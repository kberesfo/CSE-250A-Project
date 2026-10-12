import pandas as pd
from cse250a.modules.config import config

def run():
    # df = pd.read_csv(config.paths.data_dir / "air_quality" / "AirQualityUCI.csv", sep=";")
    print(config.seed)

# in the runner: save the full resolved config with each run
# (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

if __name__=="__main__":
    run()