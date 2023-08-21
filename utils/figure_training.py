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
        "/images/compare/Eval_Augmented_Holonet_Zone_Plate_0.2_200000_1000/green/summaries/events.out.tfevents.1690788301.a41e39c63788.554.0",
        "/images/compare/Eval_Augmented_Holonet_Zone_Plate_0.2_600000_100/green/summaries/events.out.tfevents.1690787533.89b898dc9c78.127.0",
        "/images/compare/Eval_Augmented_Holonet_Zone_Plate_0.2_600000_1000/green/summaries/events.out.tfevents.1690788339.1b4102e0c273.379.0",
        # "/images/compare/Eval_Augmented_Holonet_Zone_Plate_0.1_180000_100/green/summaries/",
        # "/images/compare/Eval_Augmented_Holonet_Zone_Plate_0.1_180000_1000/green/summaries/",
    ]

    names=[
        "eval_0.2_0.3_1000",
        "eval_0.2_0.5_100",
        "eval_0.2_0.5_1000",
        # "eval_0.1_1.0_100",
        # "eval_0.1_1.0_1000"
    ]

    out_array=[
        "/images/final_output",
        "/images/final_output",
        "/images/final_output",
        # "/images/final_output",
        # "/images/final_output",
    ]

    

    

    tag=[
        "PSNR_1",
        "Time_1"
    ]



    for i in range(len(log_array)):
        export_scalars_to_csv(log_array[i],out_array[i],None,names[i])
