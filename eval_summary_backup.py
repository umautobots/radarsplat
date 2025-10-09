import os
import json
import pandas as pd

# Display DataFrame (df) with numbers rounded to two decimal places and not in scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)

def load_RF_evals_from_paths(experiment_names, root_path='/home/pckung/RadarFields/workspace/imgs', eval_file="eval_render_test.json", geo_eval_file="eval_geometry_lidar_map:True_tau0.5.json"):
    rows = []
    for new_name, experiment_name in experiment_names.items():
        eval_path = os.path.join(os.path.join(root_path, experiment_name), eval_file)
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                data = json.load(f)
                data["experiment"] = new_name
        else:
            print(f"[WARNING] Missing {eval_file} in {eval_path}")

        eval_geometry_path = os.path.join(os.path.join(root_path, experiment_name), geo_eval_file)
        if os.path.exists(eval_geometry_path):
            with open(eval_geometry_path, "r") as f:
                data_geo = json.load(f)
        else:
            print(f"[WARNING] Missing {geo_eval_file} in {eval_geometry_path}")

        rows.append({**data, **data_geo})

    df = pd.DataFrame(rows)
    df = df[["experiment"] + [col for col in df.columns if col != "experiment"]]
    return df

def load_RS_evals_from_paths(experiment_names, root_path='/home/pckung/gsplat/batch_results', eval_file="stats/val_step:1999_lidarmap:True_tau:0_5.json"):
    rows = []
    for new_name, experiment_name in experiment_names.items():
        eval_path = os.path.join(os.path.join(root_path, experiment_name), eval_file)
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                data = json.load(f)
                data["experiment"] = new_name
        else:
            print(f"[WARNING] Missing {eval_file} in {eval_path}")

        rows.append(data)

    df = pd.DataFrame(rows)
    df = df[["experiment"] + [col for col in df.columns if col != "experiment"]]
    return df

def print_img_and_recon_eval(df, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_occ=False, show_time=False):
    img_eval_df = df[~df["experiment"].isin(img_eval_exclude_seqs)]
    img_eval_df = img_eval_df[["psnr", "ssim", "lpips"]]
    img_eval_mean_df = pd.DataFrame([img_eval_df.mean(numeric_only=True)])
    img_eval_mean_df.index = ["Image Eval. Mean"]
    print(img_eval_mean_df)
    recon_eval_df = df[~df["experiment"].isin(recon_eval_exclude_seqs)]
    # recon_eval_df_ = recon_eval_df[["RMSE", "R-CD", "accuracy"]]
    recon_eval_df_ = recon_eval_df[["RMSE", "R-CD", "accuracy", "precision", "recall"]]
    recon_eval_mean_df = pd.DataFrame([recon_eval_df_.mean(numeric_only=True)])
    recon_eval_mean_df.index = ["Recon. Eval. Mean"]
    print(recon_eval_mean_df)
    if show_occ:
        recon_eval_df_ = recon_eval_df[["Occ_RMSE", "Occ_R-CD", "Occ_accuracy"]]
        recon_eval_mean_df = pd.DataFrame([recon_eval_df_.mean(numeric_only=True)])
        recon_eval_mean_df.index = ["Recon. Eval. Mean"]
        print(recon_eval_mean_df)
    if show_time:
        recon_eval_df_ = recon_eval_df[["ellipse_time"]]
        recon_eval_mean_df = pd.DataFrame([recon_eval_df_.mean(numeric_only=True)])
        recon_eval_mean_df.index = ["Ellipse Time"]
        print(recon_eval_mean_df)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("path1", help="Source directory (PATH1)")
    # parser.add_argument("path2", help="Destination directory (PATH2)")
    # args = parser.parse_args()

    RF_experiment_names = {
        "seq1": "boreas-2021-09-02-11-42_frame:27-67_iter:300_2025-02-23_19-37-17",
        "seq2": "boreas-2021-09-02-11-42_frame:330-370_iter:300_2025-02-24_21-23-42",
        "seq3": "boreas-2021-09-02-11-42_frame:370-410_iter:300_2025-02-24_21-46-59",
        "seq4": "boreas-2021-09-02-11-42_frame:472-512_iter:300_2025-02-24_22-24-30",

        "seq5": "boreas-2021-04-08-12-44_frame:50-90_iter:300_2025-05-11_20-58-47",
        "seq6": "boreas-2021-04-08-12-44_frame:160-200_iter:300_2025-05-11_21-01-30",
        "seq7": "boreas-2021-04-08-12-44_frame:245-285_iter:300_2025-05-11_21-04-12",
        "seq8": "boreas-2021-04-08-12-44_frame:385-425_iter:300_2025-05-11_21-06-55",

        "snow1": "boreas-2021-01-26-11-22_frame:217-257_iter:300_2025-05-11_05-53-02",
        "snow2": "boreas-2021-01-26-11-22_frame:460-500_iter:300_2025-05-11_05-55-58",

        "rain1": "boreas-2021-04-29-15-55_frame:190-230_iter:300_2025-05-11_05-59-14",
        "rain2": "boreas-2021-04-29-15-55_frame:230-270_iter:300_2025-05-11_06-02-12",

        "night1": "boreas-2021-09-14-20-00_frame:80-120_iter:300_2025-05-11_06-05-35",
        "night2": "boreas-2021-09-14-20-00_frame:410-450_iter:300_2025-05-11_13-04-21",

        # "cloud1": "boreas-2021-10-15-12-35_frame:55-95_iter:300_2025-05-11_20-53-22",
        # "cloud2": "boreas-2021-10-15-12-35_frame:365-405_iter:300_2025-05-11_20-56-05",
    }

    RF_our_occ_experiment_names = {
        # Sunny1
        "seq1": "boreas-2021-09-02-11-42_frame:27-67_iter:300_2025-05-12_03-55-51",
        "seq2": "boreas-2021-09-02-11-42_frame:330-370_iter:300_2025-05-12_03-58-43",
        "seq3": "boreas-2021-09-02-11-42_frame:370-410_iter:300_2025-05-12_04-01-43",
        "seq4": "boreas-2021-09-02-11-42_frame:472-512_iter:300_2025-05-12_04-04-36",
        # Sunny2
        "seq5": "boreas-2021-04-08-12-44_frame:50-90_iter:300_2025-05-12_04-30-50",
        "seq6": "boreas-2021-04-08-12-44_frame:160-200_iter:300_2025-05-12_04-33-42",
        "seq7": "boreas-2021-04-08-12-44_frame:245-285_iter:300_2025-05-12_04-36-33",
        "seq8": "boreas-2021-04-08-12-44_frame:385-425_iter:300_2025-05-12_04-39-32",
        # Snow
        "snow1": "boreas-2021-01-26-11-22_frame:217-257_iter:300_2025-05-12_04-07-30",
        "snow2": "boreas-2021-01-26-11-22_frame:460-500_iter:300_2025-05-12_04-10-19",
        # Rain
        "rain1": "boreas-2021-04-29-15-55_frame:190-230_iter:300_2025-05-12_04-13-10",
        "rain2": "boreas-2021-04-29-15-55_frame:230-270_iter:300_2025-05-12_04-16-13",
        # Night
        "night1": "boreas-2021-09-14-20-00_frame:80-120_iter:300_2025-05-12_04-19-11",
        "night2": "boreas-2021-09-14-20-00_frame:410-450_iter:300_2025-05-12_04-22-01",
        # Cloudy
        # "cloud1": "boreas-2021-10-15-12-35_frame:55-95_iter:300_2025-05-12_04-24-56",
        # "cloud2": "boreas-2021-10-15-12-35_frame:365-405_iter:300_2025-05-12_04-27-50",
    }

    RS_experiment_names = {
        "seq1": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250511_031707",
        "seq2": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250511_035423",
        "seq3": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250511_044239",
        "seq4": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250511_051915",

        "seq5": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250512_005700",
        "seq6": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250512_010639",
        "seq7": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250512_011621",
        "seq8": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250512_012533",

        "snow1": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250511_052618",
        "snow2": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250511_204947",

        "rain1": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250511_205954",
        "rain2": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250512_002155",

        "night1": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250511_210949",
        "night2": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250512_001103",

        # "cloud1": "boreas-2021-10-15-12-35/boreas-2021-10-15-12-35_frame_55_95_20250512_034402",
        # "cloud2": "boreas-2021-10-15-12-35/boreas-2021-10-15-12-35_frame_365_405_20250512_040333",
    }

    RS_N5000_experiment_names = {
        'seq1': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250513_031253',
        'seq2': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250513_034033',
        'seq3': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250513_035313',
        'seq4': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250513_040645',
        'seq5': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250513_061820',
        'seq6': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250513_062838',
        'seq7': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250513_063905',
        'seq8': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250513_064930',
        'snow1': 'boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250513_043327',
        'snow2': 'boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250513_050550',
        'rain1': 'boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250513_053111',
        'rain2': 'boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250513_054411',
        'night1': 'boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250513_055711',
        'night2': 'boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250513_061009'
    }

    RS_N10000_experiment_names = {
        # Sunny1
        "seq1": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250512_041520",
        "seq2": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250512_042635",
        "seq3": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250512_043714",
        "seq4": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250512_044819",

        # Sunny2
        "seq5": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250512_235806",
        "seq6": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250512_055421",
        "seq7": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250512_060409",
        "seq8": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250512_061418",

        # Snow
        "snow1": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250512_045844",
        "snow2": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250513_020110",

        # Rain
        "rain1": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250512_051143",
        "rain2": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250512_052158",

        # Night
        "night1": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250512_053222",
        "night2": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250512_054214",

        # Cloudy
        # "cloud1": "boreas-2021-10-15-12-35/boreas-2021-10-15-12-35_frame_55_95_20250512_062348",
        # "cloud2": "boreas-2021-10-15-12-35/boreas-2021-10-15-12-35_frame_365_405_20250512_062427",
    }

    RS_N30000_experiment_names = {
        'seq1': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250513_025126',
        'seq2': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250513_030256',
        'seq3': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250513_031203',
        'seq4': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250513_034238',
        'seq5': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250513_051223',
        'seq6': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250513_053327',
        'seq7': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250513_054509',
        'seq8': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250513_055715',
        'snow1': 'boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250513_035405',
        'snow2': 'boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250513_040642',
        'rain1': 'boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250513_042608',
        'rain2': 'boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250513_043540',
        'night1': 'boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250513_045138',
        'night2': 'boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250513_050204'
    }

    RS_S01_experiment_names = {
        'seq1': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250513_091642',
        'seq2': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250513_092558',
        'seq3': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250513_093637',
        'seq4': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250513_094448',
        'seq5': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250513_105357',
        'seq6': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250513_110215',
        'seq7': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250513_112230',
        'seq8': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250513_112942',
        'snow1': 'boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250513_095127',
        'snow2': 'boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250513_100004',
        'rain1': 'boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250513_100759',
        'rain2': 'boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250513_101511',
        'night1': 'boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250513_102244',
        'night2': 'boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250513_103129'
    }

    RS_S03_experiment_names = {
        'seq1': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250513_113608',
        'seq2': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250513_114453',
        'seq3': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250513_115256',
        'seq4': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250513_120150',
        'seq5': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250513_125802',
        'seq6': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250513_130621',
        'seq7': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250513_131714',
        'seq8': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250513_132521',
        'snow1': 'boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250513_120948',
        'snow2': 'boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250513_121836',
        'rain1': 'boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250513_122711',
        'rain2': 'boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250513_123506',
        'night1': 'boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250513_124314',
        'night2': 'boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250513_125111'
    }

    RS_S07_experiment_names = {
        'seq1': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250513_211857',
        'seq2': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250513_213026',
        'seq3': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250513_230415',
        'seq4': 'boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250513_231617',
        'seq5': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250514_005225',
        'seq6': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250514_010343',
        'seq7': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250514_011419',
        'seq8': 'boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250514_012455',
        'snow1': 'boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250513_233848',
        'snow2': 'boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250514_020032',
        'rain1': 'boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250514_000807',
        'rain2': 'boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250514_002125',
        'night1': 'boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250514_003210',
        'night2': 'boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250514_004305'
    }

    RS_wo_noise_prob_experiment_names = {
        # Sunny1
        "seq1": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250513_032220",
        "seq2": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250513_033110",
        "seq3": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250513_034839",
        "seq4": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250513_035757",

        # Sunny2
        "seq5": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250513_051032",
        "seq6": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250513_051911",
        "seq7": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250513_052916",
        "seq8": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250513_053743",

        # Snow
        "snow1": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250513_040635",
        "snow2": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250513_041542",

        # Rain
        "rain1": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250513_042824",
        "rain2": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250513_043731",

        # Night
        "night1": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250513_045433",
        "night2": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250513_050301",
    }

    RS_wo_mp_modeling_experiment_names = {
        # Sunny1
        "seq1": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250513_054547",
        "seq2": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250513_055532",
        "seq3": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250513_060429",
        "seq4": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250513_061417",

        # Sunny2
        "seq5": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250513_072736",
        "seq6": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250513_073645",
        "seq7": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250513_075106",
        "seq8": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250513_080002",

        # Snow
        "snow1": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250513_062307",
        "snow2": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250513_063238",

        # Rain
        "rain1": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250513_064219",
        "rain2": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250513_065100",

        # Night
        "night1": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250513_070312",
        "night2": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250513_071201",
    }

    RS_wo_sl_experiment_names = {
        # Sunny1
        "seq1": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250513_081220",
        "seq2": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250513_082209",
        "seq3": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250513_083110",
        "seq4": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250513_084101",

        # Sunny2
        "seq5": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250513_095922",
        "seq6": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250513_101310",
        "seq7": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250513_102203",
        "seq8": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250513_103108",

        # Snow
        "snow1": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250513_085535",
        "snow2": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250513_090457",

        # Rain
        "rain1": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250513_092438",
        "rain2": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250513_093322",

        # Night
        "night1": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250513_094246",
        "night2": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250513_095148",
    }

    RS_wo_occ_experiment_names = {
        # Sunny1
        "seq1": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250513_105333",
        "seq2": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250513_112506",
        "seq3": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250513_113852",
        "seq4": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250513_115247",

        # Sunny2
        "seq5": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250513_142325",
        "seq6": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250513_143714",
        "seq7": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250513_173537",
        "seq8": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250513_192209",

        # Snow
        "snow1": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250513_123227",
        "snow2": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250513_171605",

        # Rain
        "rain1": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250513_133256",
        "rain2": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250513_134511",

        # Night
        "night1": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250513_135914",
        "night2": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250513_141109",
    }

    RS_rf_occ_experiment_names = {
        "seq1": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_27_67_20250513_215056",
        "seq2": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_330_370_20250513_230548",
        "seq3": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_370_410_20250513_231502",
        "seq4": "boreas-2021-09-02-11-42/boreas-2021-09-02-11-42_frame_472_512_20250513_232450",
        "seq5": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_50_90_20250514_005742",
        "seq6": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_160_200_20250514_010724",
        "seq7": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_245_285_20250514_011648",
        "seq8": "boreas-2021-04-08-12-44/boreas-2021-04-08-12-44_frame_385_425_20250514_012608",
        "snow1": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_217_257_20250513_233839",
        "snow2": "boreas-2021-01-26-11-22/boreas-2021-01-26-11-22_frame_460_500_20250514_000315",
        "rain1": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_190_230_20250514_002209",
        "rain2": "boreas-2021-04-29-15-55/boreas-2021-04-29-15-55_frame_230_270_20250514_003101",
        "night1": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_80_120_20250514_004027",
        "night2": "boreas-2021-09-14-20-00/boreas-2021-09-14-20-00_frame_410_450_20250514_004938"
    }

    img_eval_exclude_seqs = ["seq4"]
    recon_eval_exclude_seqs = ["seq4", "snow1", "snow2"]

    # print("------------- Radar Fields -------------")

    # rf_root_path='/mnt/ws-frb/users/frank/frank/RadarFields_experiment/workspace/imgs'
    # df_radarfields = load_RF_evals_from_paths(RF_experiment_names, root_path=rf_root_path)
    # print_img_and_recon_eval(df_radarfields, img_eval_exclude_seqs, recon_eval_exclude_seqs)
    
    # print("------------- RF w/ our occ -------------")
    # rf_occ_root_path='/mnt/ws-frb/users/frank/frank/RadarFields_experiment/workspace/imgs_our_occ_map'
    # df_radarfields_our_occ = load_RF_evals_from_paths(RF_our_occ_experiment_names, root_path=rf_occ_root_path)
    # print_img_and_recon_eval(df_radarfields_our_occ, img_eval_exclude_seqs, recon_eval_exclude_seqs)

    print("------------- RadarSplat -------------")
    df_radarsplat = load_RS_evals_from_paths(RS_experiment_names)
    print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_occ=True)

    print("------------- [Ablation] RS w/o noise probability  -------------")
    df_radarsplat = load_RS_evals_from_paths(RS_wo_noise_prob_experiment_names, root_path='/home/pckung/gsplat/batch_ablations/wo_noise_prob')
    print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs)

    print("------------- [Ablation] RS w/o multipath modeling  -------------")
    df_radarsplat = load_RS_evals_from_paths(RS_wo_mp_modeling_experiment_names, root_path='/home/pckung/gsplat/batch_ablations/wo_mp_modeling')
    print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs)

    # print("------------- [Ablation] RS w/o spectual leakage  -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_wo_sl_experiment_names, root_path='/home/pckung/gsplat/batch_ablations/wo_sl')
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs)

    # print("------------- [Ablation] RS w/o occupancy map  -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_wo_occ_experiment_names, root_path='/home/pckung/gsplat/batch_ablations/wo_occ')
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs)

    # print("------------- [Ablation] RS w/ RF occupancy map  -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_rf_occ_experiment_names, root_path='/home/pckung/gsplat/batch_ablations/rf_occ')
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_occ=True)

    # print("------------- [Num Gaussian Ablation] RS N=5000 -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_N5000_experiment_names, root_path='/home/pckung/gsplat/batch_init_ablations/N_5000')
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_time=True)

    # print("------------- [Num Gaussian Ablation] RS N=10000 -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_N10000_experiment_names, root_path='/home/pckung/gsplat/batch_init_ablations/N_10000')
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_time=True)

    # print("------------- [Num Gaussian Ablation] RS N=20000 -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_experiment_names)
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_time=True)

    # print("------------- [Num Gaussian Ablation] RS N=30000 -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_N30000_experiment_names, root_path='/home/pckung/gsplat/batch_init_ablations/N_30000')
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_time=True)

    # print("------------- [Size Gaussian Ablation] RS S=0.1 -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_S01_experiment_names, root_path='/home/pckung/gsplat/batch_init_ablations/S_0.1')
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_time=True)

    # print("------------- [Size Gaussian Ablation] RS S=0.3 -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_S03_experiment_names, root_path='/home/pckung/gsplat/batch_init_ablations/S_0.3')
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_time=True)

    # print("------------- [Size Gaussian Ablation] RS S=0.5 -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_experiment_names)
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_time=True)

    # print("------------- [Size Gaussian Ablation] RS S=0.7 -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_S07_experiment_names, root_path='/home/pckung/gsplat/batch_init_ablations/S_0.7')
    # print_img_and_recon_eval(df_radarsplat, img_eval_exclude_seqs, recon_eval_exclude_seqs, show_time=True)

    # ============================================================================
    # forest_seqs = ["seq4"]
    # city_seqs = ["seq1"]

    # print("------------- Forest RS -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_experiment_names)
    # forest_seqs_eval_df = df_radarsplat[df_radarsplat["experiment"].isin(forest_seqs)]
    # print_img_and_recon_eval(forest_seqs_eval_df, [], [])
    # print("------------- Forest RS w/o multipath modeling -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_wo_mp_modeling_experiment_names, root_path='/home/pckung/gsplat/batch_ablations/wo_mp_modeling')
    # forest_seqs_eval_df = df_radarsplat[df_radarsplat["experiment"].isin(forest_seqs)]
    # print_img_and_recon_eval(forest_seqs_eval_df, [], [])
    # print("------------- City RS -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_experiment_names)
    # city_seqs_eval_df = df_radarsplat[df_radarsplat["experiment"].isin(city_seqs)]
    # print_img_and_recon_eval(city_seqs_eval_df, [], [])
    # print("------------- City RS w/o multipath modeling -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_wo_mp_modeling_experiment_names, root_path='/home/pckung/gsplat/batch_ablations/wo_mp_modeling')
    # city_seqs_eval_df = df_radarsplat[df_radarsplat["experiment"].isin(city_seqs)]
    # print_img_and_recon_eval(city_seqs_eval_df, [], [])

    # ============================================================================
    sunny_seqs = ["seq1", "seq2", "seq3", "seq5", "seq6", "seq7", "seq8"]
    night_seqs = ["night1", "night2"]
    rain_seqs = ["rain1", "rain2"]
    snow_seqs = ["snow1", "snow2"]

    # print("------------- Sunny RF -------------")
    # df_radarfields = load_RF_evals_from_paths(RF_experiment_names)
    # df_radarfields = df_radarfields[df_radarfields["experiment"].isin(sunny_seqs)]
    # print_img_and_recon_eval(df_radarfields, [], [])
    # print("------------- Sunny RS -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_experiment_names)
    # df_radarsplat = df_radarsplat[df_radarsplat["experiment"].isin(sunny_seqs)]
    # print_img_and_recon_eval(df_radarsplat, [], [])
    
    # print("------------- Night RF -------------")
    # df_radarfields = load_RF_evals_from_paths(RF_experiment_names)
    # df_radarfields = df_radarfields[df_radarfields["experiment"].isin(night_seqs)]
    # print_img_and_recon_eval(df_radarfields, [], [])
    # print("------------- Night RS -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_experiment_names)
    # df_radarsplat = df_radarsplat[df_radarsplat["experiment"].isin(night_seqs)]
    # print_img_and_recon_eval(df_radarsplat, [], [])

    # print("------------- Rain RF -------------")
    # df_radarfields = load_RF_evals_from_paths(RF_experiment_names)
    # df_radarfields = df_radarfields[df_radarfields["experiment"].isin(rain_seqs)]
    # print_img_and_recon_eval(df_radarfields, [], [])
    # print("------------- Rain RS -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_experiment_names)
    # df_radarsplat = df_radarsplat[df_radarsplat["experiment"].isin(rain_seqs)]
    # print_img_and_recon_eval(df_radarsplat, [], [])

    # print("------------- Snow RF -------------")
    # df_radarfields = load_RF_evals_from_paths(RF_experiment_names)
    # df_radarfields = df_radarfields[df_radarfields["experiment"].isin(snow_seqs)]
    # print_img_and_recon_eval(df_radarfields, [], [])
    # print("------------- Snow RS -------------")
    # df_radarsplat = load_RS_evals_from_paths(RS_experiment_names)
    # df_radarsplat = df_radarsplat[df_radarsplat["experiment"].isin(snow_seqs)]
    # print(df_radarsplat)
    # print_img_and_recon_eval(df_radarsplat, [], [])