# -*- coding: utf-8 -*-
"""
Script to download and chunk the cimec/lambada dataset from Hugging Face.
Saves the dataset in 50MB parquet chunks to the data/raw directory.
"""
import click
import logging
import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from math import ceil


def save_dataset_in_chunks(dataset, output_dir, chunk_size_mb=50, split_name="train"):
    """
    Save dataset in chunks of specified size (MB) as parquet files.

    Args:
        dataset: HuggingFace dataset
        output_dir: Directory to save chunks
        chunk_size_mb: Size of each chunk in MB
        split_name: Name of the split (train, test, validation)
    """
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to pandas for easier chunking
    df = dataset.to_pandas()

    # Estimate rows per chunk based on size
    # Rough estimate: assume average row size and target chunk size
    sample_size = min(1000, len(df))
    sample_df = df.head(sample_size)

    # Save sample to get approximate size per row
    temp_file = output_dir / "temp_sample.parquet"
    sample_df.to_parquet(temp_file)
    sample_size_mb = temp_file.stat().st_size / (1024 * 1024)
    temp_file.unlink()  # Remove temp file

    avg_row_size_mb = sample_size_mb / sample_size
    rows_per_chunk = max(1, int(chunk_size_mb / avg_row_size_mb))

    logger.info(f"Estimated {avg_row_size_mb:.4f} MB per row")
    logger.info(f"Target chunk size: {chunk_size_mb} MB")
    logger.info(f"Rows per chunk: {rows_per_chunk}")

    # Split into chunks
    total_rows = len(df)
    num_chunks = ceil(total_rows / rows_per_chunk)

    logger.info(f"Splitting {total_rows} rows into {num_chunks} chunks")

    for i in range(num_chunks):
        start_idx = i * rows_per_chunk
        end_idx = min((i + 1) * rows_per_chunk, total_rows)

        chunk_df = df.iloc[start_idx:end_idx]
        chunk_filename = f"{split_name}_chunk_{i:04d}.parquet"
        chunk_path = output_dir / chunk_filename

        chunk_df.to_parquet(chunk_path, index=False)
        chunk_size_actual = chunk_path.stat().st_size / (1024 * 1024)

        logger.info(
            f"Saved {chunk_filename}: {len(chunk_df)} rows, {chunk_size_actual:.2f} MB"
        )

    logger.info(f"Dataset chunking complete. Saved {num_chunks} chunks to {output_dir}")


@click.command()
@click.option(
    "--dataset-name", default="cimec/lambada", help="HuggingFace dataset name"
)
@click.option("--chunk-size", default=50, help="Chunk size in MB")
@click.option(
    "--output-dir", default=None, help="Output directory (defaults to data/raw)"
)
def main(dataset_name, chunk_size, output_dir):
    """
    Download and chunk HuggingFace dataset into parquet files.

    Args:
        dataset_name: Name of the HuggingFace dataset
        chunk_size: Size of each chunk in MB
        output_dir: Directory to save chunks
    """
    logger = logging.getLogger(__name__)

    # Set up paths
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "data" / "raw"

    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target chunk size: {chunk_size} MB")

    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)
        logger.info(f"Dataset loaded successfully")
        logger.info(f"Available splits: {list(dataset.keys())}")

        # Process each split
        for split_name, split_data in dataset.items():
            logger.info(
                f"Processing split '{split_name}' with {len(split_data)} examples"
            )

            split_output_dir = output_dir / split_name
            save_dataset_in_chunks(
                split_data,
                split_output_dir,
                chunk_size_mb=chunk_size,
                split_name=split_name,
            )

        logger.info("Dataset processing completed successfully!")

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
