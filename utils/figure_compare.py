import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def export_scalars_to_csv(log_dir, output_dir, tag_filter=None):
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
        output_file = os.path.join(output_dir, f"{tag}.csv")
        df.to_csv(output_file, index=False)
        print(f"Exported scalar '{tag}' to '{output_file}'")

if __name__ == "__main__":

    log_dir="/images/comapre/Eval_Augmented_Holonet_Zone_Plate/green/summaries/events.out.tfevents.1685265889.242733d6a94e.1636.0"
    output_dir="/images/csv_phaes/Eval_Augmented_Holonet_Zone_Plate/green/summaries/green"
    tag_filter="Image_No"


    export_scalars_to_csv(log_dir,output_dir,tag_filter)
