# src/models/detector_v3_standard.py

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, Add, UpSampling2D, Concatenate, Input, Reshape, Lambda
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model
import math
import yaml
from pathlib import Path

# --- Helper function to load config ---
# (Assuming config loader utilities are available or we load directly)
def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            print(f"WARNING: Config file {config_path} content is not a dictionary.")
            return {}
        return config
    except FileNotFoundError:
        print(f"ERROR: Config file not found at {config_path}")
        return {}
    except Exception as e:
        print(f"ERROR parsing config file {config_path}: {e}")
        return {}

# --- Model Architecture ---

def build_detector_v3_standard(config):
    """
    Builds the standard object detection model with EfficientNetB0, FPN,
    and separate classification and regression heads.

    Args:
        config (dict): Configuration dictionary loaded from YAML.

    Returns:
        tf.keras.Model: The built detection model.
    """
    input_shape = config['input_shape']
    num_classes = config['num_classes']
    fpn_filters = config['fpn_filters']
    num_anchors_per_level = config['num_anchors_per_level'] # Dict e.g., {'P3': 18, 'P4': 24, 'P5': 24}
    freeze_backbone = config.get('freeze_backbone', True) # Use get with default

    # Input Layer
    input_tensor = Input(shape=input_shape, name='input_image')

    # --- Backbone: EfficientNetB0 ---
    # Load pre-trained EfficientNetB0, excluding the top classification layer
    base_model = EfficientNetB0(
        input_tensor=input_tensor,
        include_top=False,
        weights='imagenet' # Use pre-trained ImageNet weights
    )

    if freeze_backbone:
        # Freeze the backbone layers
        base_model.trainable = False
        print("EfficientNetB0 backbone frozen.")
    else:
        base_model.trainable = True
        print("EfficientNetB0 backbone is trainable.")

    # --- Feature Extraction from Backbone ---
    # We need feature maps at different strides (8, 16, 32) for FPN.
    # For EfficientNetB0, these typically come from the outputs of blocks.
    # Based on your summary for 512x512:
    # Stride 8 (64x64): 'block3b_add' (shape (None, 64, 64, 40))
    # Stride 16 (32x32): 'block4c_add' (shape (None, 32, 32, 80))
    # Stride 32 (16x16): 'block6d_add' (shape (None, 16, 16, 192))
    c3_layer_name = 'block3b_add'
    c4_layer_name = 'block4c_add'
    c5_layer_name = 'block6d_add' # Using block6d_add (192 filters) as C5 source

    try:
        c3_output = base_model.get_layer(c3_layer_name).output
        c4_output = base_model.get_layer(c4_layer_name).output
        c5_output = base_model.get_layer(c5_layer_name).output
    except ValueError as e:
        print(f"ERROR: Could not find required backbone layers. Verify layer names match base_model.summary(). Error: {e}")
        raise # Re-raise the exception

    print("Backbone feature outputs identified.")

    # --- Neck: Feature Pyramid Network (FPN) ---
    # Implement the standard FPN from Lin et al. (FPN paper)
    c3_lateral = Conv2D(fpn_filters, 1, 1, padding='same', name='fpn_c3_lateral', kernel_initializer='he_normal')(c3_output)
    c4_lateral = Conv2D(fpn_filters, 1, 1, padding='same', name='fpn_c4_lateral', kernel_initializer='he_normal')(c4_output)
    c5_lateral = Conv2D(fpn_filters, 1, 1, padding='same', name='fpn_c5_lateral', kernel_initializer='he_normal')(c5_output)

    p5_top_down = c5_lateral
    p4_top_down = Add(name='fpn_p4_add')([UpSampling2D(size=(2, 2), name='fpn_p5_upsample')(p5_top_down), c4_lateral])
    p3_top_down = Add(name='fpn_p3_add')([UpSampling2D(size=(2, 2), name='fpn_p4_upsample')(p4_top_down), c3_lateral])

    p3_output = Conv2D(fpn_filters, 3, 1, padding='same', name='fpn_p3_output', kernel_initializer='he_normal')(p3_top_down)
    p4_output = Conv2D(fpn_filters, 3, 1, padding='same', name='fpn_p4_output', kernel_initializer='he_normal')(p4_top_down)
    p5_output = Conv2D(fpn_filters, 3, 1, padding='same', name='fpn_p5_output', kernel_initializer='he_normal')(p5_top_down)

    fpn_outputs = [p3_output, p4_output, p5_output] # Order: smallest stride first (P3)
    print("FPN neck built.")


    # --- Prediction Heads (Separate for Classification and Regression) ---

    classification_outputs = []
    regression_outputs = []
    level_names = ['P3', 'P4', 'P5']

    # --- Define Lambda layers for reshaping BEFORE the loop ---
    # This prevents the loop variable capture issue.
    # Each lambda is specific to a level and captures its correct num_anchors value.

    def make_reshape_lambda_cls(level_name_arg, num_anchors_arg, num_classes_arg):
         return Lambda(
             lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], num_anchors_arg, num_classes_arg)),
             name=f'{level_name_arg}_cls_reshape'
         )

    def make_reshape_lambda_reg(level_name_arg, num_anchors_arg):
         return Lambda(
             lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], num_anchors_arg, 4)),
             name=f'{level_name_arg}_reg_reshape'
         )

    reshape_cls_lambdas = {
        'P3': make_reshape_lambda_cls('P3', num_anchors_per_level['P3'], num_classes),
        'P4': make_reshape_lambda_cls('P4', num_anchors_per_level['P4'], num_classes),
        'P5': make_reshape_lambda_cls('P5', num_anchors_per_level['P5'], num_classes)
    }

    reshape_reg_lambdas = {
        'P3': make_reshape_lambda_reg('P3', num_anchors_per_level['P3']),
        'P4': make_reshape_lambda_reg('P4', num_anchors_per_level['P4']),
        'P5': make_reshape_lambda_reg('P5', num_anchors_per_level['P5'])
    }

    # --- Now, iterate through levels and build heads, applying the correct pre-defined Lambda ---
    for i, level_output in enumerate(fpn_outputs):
        level_name = level_names[i]
        current_level_num_anchors = num_anchors_per_level[level_name] # Use this for Conv2D filter count

        # Common layers for heads (e.g., 4 convolutional layers)
        x_cls = level_output
        x_reg = level_output

        # Build shared head layers (example: 4 Conv+BN+ReLU blocks)
        for j in range(4): # Standard number of layers in head
            x_cls = Conv2D(fpn_filters, 3, 1, padding='same', name=f'cls_head_{level_name}_conv{j}', kernel_initializer='he_normal')(x_cls)
            x_cls = BatchNormalization(name=f'cls_head_{level_name}_bn{j}')(x_cls)
            x_cls = ReLU(name=f'cls_head_{level_name}_relu{j}')(x_cls)

            x_reg = Conv2D(fpn_filters, 3, 1, padding='same', name=f'reg_head_{level_name}_conv{j}', kernel_initializer='he_normal')(x_reg)
            x_reg = BatchNormalization(name=f'reg_head_{level_name}_bn{j}')(x_reg)
            x_reg = ReLU(name=f'reg_head_{level_name}_relu{j}')(x_reg)


        # Final Classification Head layer
        cls_filters = current_level_num_anchors * num_classes
        classification_output_conv = Conv2D( # Renamed to avoid confusion with reshaped output
            cls_filters,
            3, 1, padding='same',
            name=f'{level_name}_cls_output_conv',
            kernel_initializer='he_normal',
            bias_initializer=tf.keras.initializers.Constant(math.log((1 - 0.01) / 0.01))
        )(x_cls)

        # Reshape classification output using the specific Lambda layer for this level
        classification_output_reshaped = reshape_cls_lambdas[level_name](classification_output_conv)
        classification_outputs.append(classification_output_reshaped)


        # Final Regression Head layer
        reg_filters = current_level_num_anchors * 4
        regression_output_conv = Conv2D( # Renamed to avoid confusion
            reg_filters,
            3, 1, padding='same',
            name=f'{level_name}_reg_output_conv',
            kernel_initializer='he_normal',
            activation='linear'
        )(x_reg)

        # Reshape regression output using the specific Lambda layer for this level
        regression_output_reshaped = reshape_reg_lambdas[level_name](regression_output_conv)
        regression_outputs.append(regression_output_reshaped)

    print("Prediction heads built.")

    # --- Assemble Final Model ---
    # The model outputs a list of tensors: [reg_P3, cls_P3, reg_P4, cls_P4, reg_P5, cls_P5]
    outputs = []
    for i in range(len(fpn_outputs)):
         outputs.append(regression_outputs[i])
         outputs.append(classification_outputs[i])

    model = Model(inputs=input_tensor, outputs=outputs, name=config.get('model_name', 'StandardDetector'))

    print(f"Model '{model.name}' built successfully.")

    return model

# --- Test Block ---
if __name__ == '__main__':
    print("--- Testing model building ---")

    # Define a dummy config for testing purposes
    # The 'num_anchors_per_level' values here MUST match what you put in your
    # actual detector_config_v3_standard.yaml after running calculate_anchors.py (e.g. K*3 ratios)
    test_config = {
        'input_shape': [512, 512, 3],
        'num_classes': 2, # pit, crack
        'fpn_filters': 256,
        # Example values based on our discussion (e.g., K=6 for P3, K=8 for P4/P5, and 3 ratios [0.5, 1.0, 2.0])
        'num_anchors_per_level': {
            'P3': 18, # Example: 6 scales * 3 ratios
            'P4': 24, # Example: 8 scales * 3 ratios
            'P5': 24  # Example: 8 scales * 3 ratios
        },
        'freeze_backbone': True, # Set to False to test trainable model
        'model_name': 'TestDetector'
        # Add any other necessary config keys with dummy values if build_detector_v3_standard requires them
    }

    # For testing, let's also check for potential EffNet layer names directly
    print("\n--- Verifying EfficientNetB0 Layer Names for Strides ---")
    temp_base_model = EfficientNetB0(input_shape=test_config['input_shape'], include_top=False, weights='imagenet')
    print("EfficientNetB0 Summary (for layer name verification):")
    temp_base_model.summary()

    # Based on the summary, verify these layer names and their output shapes:
    # Stride 8 (64x64): 'block3b_add' (shape (None, 64, 64, 40))
    # Stride 16 (32x32): 'block4c_add' (shape (None, 32, 32, 80))
    # Stride 32 (16x16): 'block6d_add' (shape (None, 16, 16, 192))
    # The layer names 'block3b_add', 'block4c_add', 'block6d_add' seem correct for strides 8, 16, 32 respectively with 512x512 input.
    # The code uses these names.

    print("\nManually verify the layer names 'block3b_add', 'block4c_add', 'block6d_add' in the summary above correspond to spatial dimensions 64x64, 32x32, 16x16 respectively for 512x512 input.")
    print("The code currently uses these names. If your summary showed different names for these output shapes, update c3_layer_name, c4_layer_name, c5_layer_name in build_detector_v3_standard.")


    try:
        # Build the model using the test config
        model = build_detector_v3_standard(test_config)

        # Print model summary
        print("\nStandard Detector Model Summary:")
        model.summary()

        # Create dummy input data with the specified input shape and batch size 1
        dummy_input_batch = tf.random.normal(shape=(1, *test_config['input_shape']), dtype=tf.float32) # Use float32

        # Get model outputs
        outputs = model(dummy_input_batch)

        # Print shapes of the output tensors
        print("\n--- Output Tensor Shapes ---")
        # Expected order: reg P3, cls P3, reg P4, cls P4, reg P5, cls P5
        level_names = ['P3', 'P4', 'P5']
        level_strides = [8, 16, 32]
        tasks = ['reg', 'cls']

        output_idx = 0
        for i, level_name in enumerate(level_names):
            # Calculate expected spatial dimensions based on input shape and stride
            expected_H = test_config['input_shape'][0] // level_strides[i]
            expected_W = test_config['input_shape'][1] // level_strides[i]
            num_anchors = test_config['num_anchors_per_level'][level_name]

            # Regression output shape
            expected_reg_shape = (1, expected_H, expected_W, num_anchors, 4)
            print(f"{level_name}_{tasks[0]} shape: {outputs[output_idx].shape} (Expected: {expected_reg_shape})")
            # Use tuple conversion for comparison
            assert tuple(outputs[output_idx].shape) == expected_reg_shape, f"Shape mismatch for {level_name}_{tasks[0]}"
            output_idx += 1

            # Classification output shape
            expected_cls_shape = (1, expected_H, expected_W, num_anchors, test_config['num_classes'])
            print(f"{level_name}_{tasks[1]} shape: {outputs[output_idx].shape} (Expected: {expected_cls_shape})")
            # Use tuple conversion for comparison
            assert tuple(outputs[output_idx].shape) == expected_cls_shape, f"Shape mismatch for {level_name}_{tasks[1]}"
            output_idx += 1

        print("\nModel building and output shape test successful!")

    except Exception as e:
        print(f"\n--- Error during model building or testing ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging