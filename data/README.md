# Data Directory

This directory contains the datasets for fake review detection.

## Structure

- `raw/`: Raw, unprocessed data files
- `processed/`: Preprocessed and split data files

## Expected Data Format

### Raw Data (`raw/reviews.csv`)

The raw data file should be a CSV with at least the following columns:

- `text` (required): The review text
- `label` (required): 0 for genuine reviews, 1 for fake reviews

Optional columns for additional features:
- `rating`: Numerical rating (e.g., 1-5)
- `reviewer_id`: Unique identifier for the reviewer
- `product_id`: Product identifier
- `timestamp`: Review timestamp (ISO format)
- `helpful_votes`: Number of helpful votes
- `verified_purchase`: Boolean indicating if purchase was verified

### Example

```csv
text,label,rating,reviewer_id,product_id,timestamp,helpful_votes,verified_purchase
"Great product! Highly recommend it.",0,5,user123,prod456,2023-01-15T10:30:00Z,10,true
"Worst purchase ever. Complete waste of money.",1,1,user789,prod456,2023-01-16T14:20:00Z,0,false
```

## Processed Data

After running `make preprocess` or `python scripts/preprocess_data.py`, the following files will be created:

- `processed/train.csv`: Training set (default 80%)
- `processed/val.csv`: Validation set (default 10%)
- `processed/test.csv`: Test set (default 10%)

## Data Collection

To use this system, you need to provide your own dataset. Some potential sources:

1. **Amazon Reviews**: Download from Amazon Review Dataset
2. **Yelp Reviews**: Available through Yelp Open Dataset
3. **Custom Datasets**: Create your own labeled dataset

**Note**: This repository does not include any review data. You must provide your own dataset.
