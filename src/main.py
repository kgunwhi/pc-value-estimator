import scraper as scraper
import preproc
import eda
import os
import cat


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "model"), exist_ok=True)
    print("====== Scraping Data ======")
    scraper.scrape_passmark_cpu(project_root)
    scraper.scrape_passmark_gpu(project_root)

    print("====== Loading Raw Data ======")
    cpu_df, gpu_df = preproc.load_data(project_root)
    print("Raw Data has been loaded")
    eda.run_full_eda(cpu_df, gpu_df, project_root, label="preclean")

    print("====== Cleaning Raw Data ======")
    cpu_df_clean, gpu_df_clean = preproc.clean_data(cpu_df, gpu_df, project_root)
    eda.run_full_eda(cpu_df_clean, gpu_df_clean, project_root, label="postclean")

    print("====== Training CPU model ======")
    cat.catboost_train_cpu(cpu_df_clean, project_root)
    print("====== Training GPU model ======")
    cat.catboost_train_gpu(gpu_df_clean, project_root)
