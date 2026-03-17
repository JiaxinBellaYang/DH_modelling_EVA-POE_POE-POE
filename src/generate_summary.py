import os
import pandas as pd
import numpy as np

def generate_full_summary():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "results")
    
    models = ["0peak", "flatpeak", "1peak", "2peak", "3peak"]
    datasets = ["Before UV", "After UV"] # Standardized names
    
    all_data = []
    
    for model in models:
        model_dir = os.path.join(results_dir, model)
        
        dit_path = os.path.join(model_dir, "dit_fit_summary.xlsx")
        uv_path = os.path.join(model_dir, f"uv_fit_summary_{model}.xlsx")
        
        if not os.path.exists(dit_path) or not os.path.exists(uv_path):
            continue
            
        dit_df = pd.read_excel(dit_path)
        uv_df = pd.read_excel(uv_path)
        
        # We know idx=0 is Before UV, idx=1 is After UV
        for idx, (dit_target, uv_target) in enumerate([("Initial", "Before"), ("UV", "After")]):
            # Find Dit goodness
            dit_row = dit_df[dit_df["Dataset"].astype(str).str.contains(dit_target, case=False, na=False)]
            
            # Find UV params and lifetime goodness
            uv_row = uv_df[uv_df["Dataset"].astype(str).str.contains(uv_target, case=False, na=False)]
            
            if dit_row.empty or uv_row.empty:
                continue
                
            dit_r = dit_row.iloc[0]
            uv_r = uv_row.iloc[0]
            
            rec = {
                "Model": model,
                "State": datasets[idx], # "Before UV" or "After UV"
                "R2_Dit": dit_r.get("R2", np.nan),
                "SSR_Dit": dit_r.get("SSR", np.nan),
                "R2_tau": uv_r.get("R2", np.nan),
                "SSR_tau": uv_r.get("SSR", np.nan),
                "Qfix_fit_1e13": uv_r.get("Qfix (cm^-2)", np.nan) / 1e13,
                
                # Baseline Dit Parameters shared
                "Dit0_v": uv_r.get("Dit0_v", np.nan),
                "Ev_trap": uv_r.get("Ev_trap", np.nan),
                "Dit0_c": uv_r.get("Dit0_c", np.nan),
                "Ec_trap": uv_r.get("Ec_trap", np.nan),
            }
            
            if model == "0peak":
                pass
            elif model == "flatpeak":
                rec["Dit0_const"] = uv_r.get("Dit0_const", np.nan)
            elif model == "1peak":
                rec["Dit0_g1"] = uv_r.get("Dit0_g", np.nan)
                rec["E0_1"] = uv_r.get("E0", np.nan)
                rec["sigma_1"] = uv_r.get("sigma", np.nan)
            elif model == "2peak":
                rec["Dit0_g1"] = uv_r.get("Dit0_g1", np.nan)
                rec["E0_1"] = uv_r.get("E0_1", np.nan)
                rec["sigma_1"] = uv_r.get("sigma_1", np.nan)
                rec["Dit0_g2"] = uv_r.get("Dit0_g2", np.nan)
                rec["E0_2"] = uv_r.get("E0_2", np.nan)
                rec["sigma_2"] = uv_r.get("sigma_2", np.nan)
            elif model == "3peak":
                rec["Dit0_g1"] = uv_r.get("Dit0_g1", np.nan)
                rec["E0_1"] = uv_r.get("E0_1", np.nan)
                rec["sigma_1"] = uv_r.get("sigma_1", np.nan)
                rec["Dit0_g2"] = uv_r.get("Dit0_g2", np.nan)
                rec["E0_2"] = uv_r.get("E0_2", np.nan)
                rec["sigma_2"] = uv_r.get("sigma_2", np.nan)
                rec["Dit0_g3"] = uv_r.get("Dit0_g3", np.nan)
                rec["E0_3"] = uv_r.get("E0_3", np.nan)
                rec["sigma_3"] = uv_r.get("sigma_3", np.nan)
                
            all_data.append(rec)
            
    if all_data:
        df_out = pd.DataFrame(all_data)
        out_path = os.path.join(results_dir, "master_model_comparison_summary.xlsx")
        df_out.to_excel(out_path, index=False)
        print(f"Generated master summary table at: {out_path}")

if __name__ == "__main__":
    generate_full_summary()
