import scraper as scraper
import preproc
import eda
import xgb
import os



if __name__ == "__main__":
    """
    Entry point for running PassMark data scraping.
    Calls functions to extract CPU and GPU benchmark scores
    and writes them to CSV files under the /data directory.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scraper.scrape_passmark_cpu(project_root)
    scraper.scrape_passmark_gpu(project_root)

    cpu_df, gpu_df = preproc.clean_data()
    #eda.plot_score_and_price_distributions(cpu_df, gpu_df)
    #eda.plot_price_vs_performance(cpu_df, gpu_df)
    #eda.show_top_value_components(cpu_df, gpu_df, top_n=10)
    print("====== Training CPU model ======")
    xgb.xgboost_train_cpu(cpu_df,project_root)
    print("====== Training GPU model ======")
    xgb.xgboost_train_gpu(gpu_df,project_root)


