import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def export_scalars_to_csv(log_dir, output_dir, tag_filters=None, name="hoge"):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    os.makedirs(output_dir, exist_ok=True)

    all_tags = event_acc.Tags()["scalars"]

    print("all_tags", len(all_tags))
    if not tag_filters:
        tags_to_export = all_tags
    else:
         tags_to_export = [tag for tag in all_tags if any(tag_filter in tag for tag_filter in tag_filters)]


    data_frames = []
    for tag in tags_to_export:
        events = event_acc.Scalars(tag)
        summaries = [{"step": e.step, f"{tag}_value": e.value} for e in events]
        df = pd.DataFrame(summaries).set_index("step")
        data_frames.append(df.reset_index())

    merged_df = pd.concat(data_frames, axis=1).rename(columns={'index': 'step'})

    output_file = os.path.join(output_dir, f"{name}.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"Exported {len(tags_to_export)} scalar(s) to '{output_file}'")

if __name__ == "__main__":
    # out_array=[
    #     "/images/csv_phaes/Train_Augmented_Holonet_Reflect_Changed_Phase/green",
    #     "/images/csv_phaes/Train_Augmented_Conditional_Unet_Zone_Plate/green",
    #     "/images/csv_phaes/Train_Augmented_Conditional_Unet_Reflect_Changed_Phase/green" 
    # ]

    log_array=[
        "/images/comapre/Eval_Augmented_Conditional_Unet_Reflect_Changed_Phase/green/summaries/events.out.tfevents.1687150958.1b4102e0c273.524.0",
        "/images/comapre/Eval_Augmented_Conditional_Unet_Zone_Plate/green/summaries/events.out.tfevents.1686967032.89b898dc9c78.852.0",
        "/images/comapre/Eval_Augmented_Holonet_Reflect_Changed_Phase/green/summaries/events.out.tfevents.1686966971.1b4102e0c273.350.0",
        "/images/comapre/Eval_Augmented_Holonet_Zone_Plate/green/summaries/events.out.tfevents.1686966819.839d54aa5f67.1353.0"
    ]

    # log_array=[
    #     "/images/comapre/GS/green/summaries/events.out.tfevents.1686647250.89b898dc9c78.730.0"
    # ]

    out_array=[
        "/images/final_output",
        "/images/final_output",
        "/images/final_output",
        "/images/final_output"
    ]

    # out_array=[
   
    # ]

    # log_array=[
     
    # ]
    
    names=[
        "eavl_unet_shift",
        "eval_unet_zone",
        "eval_holo_shift",
        "eval_holo_zone"
    ]

    # tag=[
    #     "PSNR_1",
    #     "PSNR_101",
    #     "PSNR_201",
    #     "PSNR_301",
    #     "PSNR_401",
    #     "PSNR_501",
    #     "PSNR_601",
    #     "PSNR_701",
    #     "PSNR_801",
    #     "PSNR_901",
    #     "PSNR_1001",
    #     "Time_1",
    #     "Time_101",
    #     "Time_201",
    #     "Time_301",
    #     "Time_401",
    #     "Time_501",
    #     "Time_601",
    #     "Time_701",
    #     "Time_801",
    #     "Time_901",
    #     "Time_1001",
    # ]

    # names=[
    #     "train_holo_zone",
    #     "train_holo_shift",
    #     "train_cunet_zone",
    #     "train_cunet_shift"
    # ]

    for i in range(len(log_array)):
        export_scalars_to_csv(log_array[i],out_array[i],None,names[i])
