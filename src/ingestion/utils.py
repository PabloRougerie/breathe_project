from pathlib import Path

def save_data_local(df, output_path):
    """
    Save DataFrame to local CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Output file path
    
    Returns:
        None
    """
    # Create parent directories if they don't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} rows to {output_path}")
