import os
import cv2
import torch
import logging
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


def setup_logger(log_file):
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Logger initialized.")


def log_tiff_info(src):
    """
    Log basic information about the TIFF file.

    Args:
        src (rasterio.DatasetReader): Opened rasterio dataset.
    """
    logging.info("Loaded GeoTIFF information:")
    logging.info("  Filepath: %s", src.name)
    logging.info("  Number of Channels: %d", src.count)
    logging.info("  Width: %d pixels", src.width)
    logging.info("  Height: %d pixels", src.height)
    logging.info("  CRS: %s", src.crs)
    logging.info("  Transform: %s", src.transform)


def load_midas_model(model_type="DPT_Large"):
    """
    Load the MiDaS model and transformation pipeline.

    Args:
        model_type (str): Type of MiDaS model to load.

    Returns:
        model (torch.nn.Module): Loaded MiDaS model.
        transform (Callable): Transformation pipeline for the model.
        device (torch.device): Computation device (CPU/GPU).
    """
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        logging.info("MiDaS model %s loaded successfully.", model_type)
        return model, transform, device
    except Exception as e:
        logging.error("Failed to load MiDaS model: %s", e)
        raise


def predict_depth(model, transform, device, image):
    """
    Predict the depth map for a single RGB image using the MiDaS model.

    Args:
        model (torch.nn.Module): MiDaS model.
        transform (Callable): Transformation pipeline.
        device (torch.device): Computation device.
        image (numpy.ndarray): Input RGB image in HxWxC format.

    Returns:
        numpy.ndarray: Normalized depth map.
    """
    input_batch = transform(image).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bilinear",
            align_corners=True,
        ).squeeze().cpu().numpy()
    return (prediction - prediction.min()) / (prediction.max() - prediction.min())


def create_weight_map(window_size):
    """
    Create a weight map for blending overlapping patches, with higher weights at the center.

    Args:
        window_size (int): Size of the patch.

    Returns:
        numpy.ndarray: Weight map of size (window_size, window_size).
    """
    return np.outer(np.hanning(window_size), np.hanning(window_size))


def process_image_with_overlap(model, transform, device, input_image, window_size, overlap):
    """
    Process an input image with overlapping patches to generate a seamless depth map.

    Args:
        model (torch.nn.Module): MiDaS model.
        transform (Callable): Transformation pipeline.
        device (torch.device): Computation device.
        input_image (numpy.ndarray): Input RGB image in HxWxC format.
        window_size (int): Size of the patches to process.
        overlap (float): Overlap ratio between patches (0 to 1).

    Returns:
        numpy.ndarray: Depth map for the input image.
    """
    height, width, _ = input_image.shape
    stride = int(window_size * (1 - overlap))
    output_depth_map = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    weight_patch = create_weight_map(window_size)

    logging.info("Processing image of size (height: %d, width: %d)", height, width)

    for y in tqdm(range(0, height, stride), desc="Processing Rows"):
        for x in range(0, width, stride):
            patch = input_image[y:y + window_size, x:x + window_size]
            if patch.shape[:2] != (window_size, window_size):
                continue

            logging.debug("Processing patch at (y: %d, x: %d)", y, x)
            depth_patch = predict_depth(model, transform, device, patch)
            output_depth_map[y:y + window_size, x:x + window_size] += depth_patch * weight_patch
            weight_map[y:y + window_size, x:x + window_size] += weight_patch

    logging.info("Depth map processing completed.")
    return output_depth_map / np.maximum(weight_map, 1e-5)


def show_orthophoto_and_depth_map(orthophoto, depth_map):
    """
    Display the orthophoto and depth map side by side using matplotlib.

    Args:
        orthophoto (numpy.ndarray): The input orthophoto image.
        depth_map (numpy.ndarray): The depth map to display.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Orthophoto
    axes[0].imshow(orthophoto / 255.0)  # Normalize for display
    axes[0].set_title("Orthophoto")
    axes[0].axis("off")

    # Depth map
    im = axes[1].imshow(depth_map, cmap="viridis")
    axes[1].set_title("Depth Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def main(args):
    """
    Main function to process the GeoTIFF and generate a depth map.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    log_file = f"{Path(__file__).stem}.log"
    setup_logger(log_file)

    model, transform, device = load_midas_model(args.model_type)

    try:
        with rasterio.open(args.input_file) as src:
            log_tiff_info(src)

            profile = src.profile
            profile.update(driver="GTiff", dtype=rasterio.float32, count=1, compress="lzw")

            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            input_image = np.transpose(src.read([1, 2, 3]), (1, 2, 0))  # Read and format RGB channels

            logging.info("Input file %s read successfully.", args.input_file)

            depth_map = process_image_with_overlap(
                model, transform, device, input_image, args.window_size, args.overlap
            )

            if args.show_output:
                show_orthophoto_and_depth_map(input_image, depth_map)

            with rasterio.open(args.output_file, "w", **profile) as dst:
                dst.write(depth_map, 1)
            logging.info("Depth map saved successfully to %s.", args.output_file)
            print(f"Depth map processing complete. File exported to: {args.output_file}")

    except Exception as e:
        logging.error("Error processing GeoTIFF: %s", e)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate depth map from a single GeoTIFF using MiDaS model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input GeoTIFF file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output depth map file.")
    parser.add_argument("--model_type", type=str, default="DPT_Large", help="MiDaS model type (default: DPT_Large).")
    parser.add_argument("--window_size", type=int, default=512, help="Size of the sliding window (pixels).")
    parser.add_argument("--overlap", type=float, default=0.7, help="Overlap between adjacent patches (0 to 1).")
    parser.add_argument("--show_output", action="store_true", help="Display the orthophoto and depth map side by side.")
    args = parser.parse_args()

    main(args)
