import scraper as scraper
import preproc

if __name__ == "__main__":
    """
    Entry point for running PassMark data scraping.
    Calls functions to extract CPU and GPU benchmark scores
    and writes them to CSV files under the /data directory.
    """
    #scraper.scrape_passmark_cpu()
    #scraper.scrape_passmark_gpu()

    cpu_df, gpu_df = preproc.load_data()
    preproc.plot_cpu_distribution(cpu_df)

