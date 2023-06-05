import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def export_scalars_to_csv(log_dir, output_dir, tag_filter=None,name="hoge"):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    os.makedirs(output_dir, exist_ok=True)

    all_tags = event_acc.Tags()["scalars"]

    print("all_tags",len(all_tags))
    if not tag_filter:
        tags_to_export = all_tags
    else:
        tags_to_export = [tag for tag in all_tags if tag_filter in tag]

    summaries = {}
    for tag in tags_to_export:
        events = event_acc.Scalars(tag)
        summaries[tag] = [{"wall_time": e.wall_time, "step": e.step, "value": e.value} for e in events]
        df = pd.DataFrame(summaries[tag])
        output_file = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(output_file, index=False)
        print(f"Exported scalar '{tag}' to '{output_file}'")

if __name__ == "__main__":
    # log_array=[
    #     "/images/phases/Train_Augmented_Holonet_Reflect Changed Phase/green/summaries/events.out.tfevents.1683428964.0198b4a1398e.10145.0",
    #     "/images/phases/Train_Augmented Conditional Unet_Zone_Plate/green/summaries/events.out.tfevents.1683428980.0198b4a1398e.10188.0",
    #     "/images/phases/Train_Augmented Conditional Unet_Reflect Changed Phase/green/summaries/events.out.tfevents.1683429011.2bfa3860b79c.6692.0"
    # ]

    # out_array=[
    #     "/images/csv_phaes/Train_Augmented_Holonet_Reflect_Changed_Phase/green",
    #     "/images/csv_phaes/Train_Augmented_Conditional_Unet_Zone_Plate/green",
    #     "/images/csv_phaes/Train_Augmented_Conditional_Unet_Reflect_Changed_Phase/green"
        
    # ]

    log_array=[
        "/images/phases/Train_Augmented Conditional Unet_Reflect Changed Phase/green/summaries/events.out.tfevents.1683429011.2bfa3860b79c.6692.0",
        "/images/phases/Train_Augmented Conditional Unet_Zone_Plate/green/summaries/events.out.tfevents.1685330450.0198b4a1398e.10610.0",
        "/images/phases/Train_Augmented_Holonet_Reflect Changed Phase/green/summaries/events.out.tfevents.1685266270.0198b4a1398e.10525.0",
        "/images/phases/Train_Augmented_Holonet_Zone_Plate/green/summaries/events.out.tfevents.1685266210.2bfa3860b79c.6899.0"
    ]

    out_array=[
        "/images/final_output",
        "/images/final_output",
        "/images/final_output",
        "/images/final_output"
    ]

    names=[
        "train_holo_zone",
        "train_holo_shift",
        "train_cunet_zone",
        "train_cunet_shift"
    ]
   
    tag_filter="Val PSNR"

    for i in range(len(log_array)):
        export_scalars_to_csv(log_array[i],out_array[i],tag_filter,names[i])
