import scraper as scraper
import preproc
import eda
import xgb
import os



if __name__ == "__main__":
    # Create model directory if missing
    os.makedirs("model", exist_ok=True)
    """
    Entry point for running PassMark data scraping.
    Calls functions to extract CPU and GPU benchmark scores
    and writes them to CSV files under the /data directory.
    """
    scraper.scrape_passmark_cpu()
    scraper.scrape_passmark_gpu()

    cpu_df, gpu_df = preproc.clean_data()
    #eda.plot_score_and_price_distributions(cpu_df, gpu_df)
    #eda.plot_price_vs_performance(cpu_df, gpu_df)
    #eda.show_top_value_components(cpu_df, gpu_df, top_n=10)

    xgb.xgboost_train_cpu(cpu_df)

