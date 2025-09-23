import os

yaml_path = r"D:\aiml project\hackthon\dataset\dataset.yaml"

if os.path.exists(yaml_path):
    print(f"✅ dataset.yaml FOUND at: {yaml_path}")
else:
    print(f"❌ dataset.yaml MISSING at: {yaml_path}")
