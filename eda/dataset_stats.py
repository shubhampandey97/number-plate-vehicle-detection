import glob
import os

def get_stats(train=True, save_file=None):
    if train:
        img_dir = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images\train'
        dataset_type = "Training"
    else:
        img_dir = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\data\final_merged_datasets\images\val'
        dataset_type = "Validation"

    class_names = ['Bike', 'Bus', 'Car', 'Number Plate', 'Person', 'Truck', 'Auto']
    hist = [0, 0, 0, 0, 0, 0, 0]

    num_images = 0

    # Match both .jpg and .png files
    image_files = glob.glob(os.path.join(img_dir, '*.jpg')) + \
                  glob.glob(os.path.join(img_dir, '*.png'))

    output_lines = []
    output_lines.append(f"{dataset_type} set stats\n")

    for image_file in image_files:
        label_file = image_file.replace('images', 'labels') \
                               .replace('.jpg', '.txt') \
                               .replace('.png', '.txt')

        if not os.path.exists(label_file):
            continue

        num_images += 1

        with open(label_file, 'r') as f:
            label_lines = f.readlines()

            for label_line in label_lines:
                label = int(label_line[0])
                hist[label] += 1

    output_lines.append(f"Number of images: {num_images}")

    for idx, name in enumerate(class_names):
        output_lines.append(f"{name}: {hist[idx]}")

    # Print output
    for line in output_lines:
        print(line)

    # Save output to file
    if save_file:
        with open(save_file, 'a') as f:
            f.write("\n".join(output_lines))
            f.write("\n\n")


# Output file path
output_path = r'D:\Guvi\Projects\major\number-plate-vehicle-detection\eda_report\dataset_stats.txt'

# Clear old file if exists
open(output_path, 'w').close()

# Save training stats
get_stats(True, output_path)

print("\n" + "="*50 + "\n")

# Save validation stats
get_stats(False, output_path)

print(f"\nStats saved to: {output_path}")