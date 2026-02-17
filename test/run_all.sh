#!/bin/bash
################################################################################
# Complete Gesture Recognition Training Pipeline
# Run this script to automate the entire training process
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "GESTURE RECOGNITION TRAINING PIPELINE"
echo "================================================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="ipn_processed"
AUGMENTED_DIR="${DATA_DIR}/balanced/augmented"
MODEL_TYPE="ultra"  # or "temporal"
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001

# Step 1: Check if IPN dataset is downloaded
echo "================================================================================"
echo "STEP 1: CHECKING DATASET"
echo "================================================================================"
echo ""

if [ ! -d "ipn_hand_dataset/frames" ]; then
    echo -e "${RED}❌ IPN Hand dataset not found${NC}"
    echo ""
    echo "Please download the dataset first:"
    echo "  python download_ipn.py"
    echo ""
    exit 1
else
    echo -e "${GREEN}✓ IPN Hand dataset found${NC}"
fi

# Step 2: Process dataset
echo ""
echo "================================================================================"
echo "STEP 2: PROCESSING DATASET"
echo "================================================================================"
echo ""

if [ ! -d "${DATA_DIR}" ]; then
    echo "Processing IPN Hand videos to extract keypoints..."
    echo "This may take 1-3 hours depending on your system."
    echo ""
    read -p "Process dataset now? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python process_ipn_hand.py \
            --ipn_dir ipn_hand_dataset/frames \
            --output_dir ${DATA_DIR} \
            --sequence_length 10 \
            --balance
        
        echo -e "${GREEN}✓ Dataset processed${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping dataset processing${NC}"
        echo "Run manually: python process_ipn_hand.py --ipn_dir ipn_hand_dataset/frames --output_dir ${DATA_DIR} --balance"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Processed dataset already exists${NC}"
fi

# Step 3: Visualize dataset (optional)
echo ""
echo "================================================================================"
echo "STEP 3: VISUALIZING DATASET (OPTIONAL)"
echo "================================================================================"
echo ""

read -p "Visualize dataset distribution? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python visualize_ipn_data.py \
        --data_dir ${DATA_DIR}/balanced \
        --action distribution
fi

# Step 4: Augment dataset
echo ""
echo "================================================================================"
echo "STEP 4: AUGMENTING DATASET"
echo "================================================================================"
echo ""

if [ ! -d "${AUGMENTED_DIR}" ]; then
    echo "Augmenting dataset (3x)..."
    python augment_ipn_data.py \
        --data_dir ${DATA_DIR}/balanced \
        --factor 3
    
    echo -e "${GREEN}✓ Dataset augmented${NC}"
else
    echo -e "${GREEN}✓ Augmented dataset already exists${NC}"
fi

# Step 5: Train model
echo ""
echo "================================================================================"
echo "STEP 5: TRAINING MODEL"
echo "================================================================================"
echo ""

echo "Training configuration:"
echo "  Model: ${MODEL_TYPE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train.py \
        --model ${MODEL_TYPE} \
        --data_dir ${AUGMENTED_DIR} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LEARNING_RATE}
    
    echo -e "${GREEN}✓ Training complete${NC}"
    
    # Find the latest run directory
    LATEST_RUN=$(ls -td runs/*/ | head -1)
    echo ""
    echo "Model saved to: ${LATEST_RUN}"
    
    # Step 6: Export to ONNX
    echo ""
    echo "================================================================================"
    echo "STEP 6: EXPORTING TO ONNX"
    echo "================================================================================"
    echo ""
    
    read -p "Export model to ONNX? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python export_onnx.py ${LATEST_RUN}/best_model.pth
        echo -e "${GREEN}✓ ONNX export complete${NC}"
        
        ONNX_FILE="${LATEST_RUN}/model.onnx"
        METADATA_FILE="${LATEST_RUN}/model.json"
        
        echo ""
        echo "================================================================================"
        echo "NEXT STEPS FOR JETSON NANO DEPLOYMENT"
        echo "================================================================================"
        echo ""
        echo "1. Transfer these files to Jetson Nano:"
        echo "   - ${ONNX_FILE}"
        echo "   - ${METADATA_FILE}"
        echo ""
        echo "2. On Jetson Nano, convert to TensorRT:"
        echo "   python convert_to_tensorrt.py model.onnx --fp16"
        echo ""
        echo "3. Run real-time inference:"
        echo "   python jetson_inference.py --engine model.trt"
        echo ""
    fi
else
    echo -e "${YELLOW}⚠ Skipping training${NC}"
    echo "Run manually: python train.py --model ${MODEL_TYPE} --data_dir ${AUGMENTED_DIR}"
fi

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "To monitor training (in a separate terminal):"
echo "  tensorboard --logdir runs/"
echo ""