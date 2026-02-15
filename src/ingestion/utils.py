def save_data_local(df, output_path):
    df.to_csv(output_path, index= False)
    print(f"✅ Saved to {output_path}")
