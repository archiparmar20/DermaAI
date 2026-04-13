# Automated Kaggle Dataset Builder

Complete automated pipeline to download, organize, and balance skin cancer datasets from Kaggle.

## 🎯 Features

✓ **Automatic Kaggle API Setup** - Detects kaggle.json, moves to ~/.kaggle/, sets permissions  
✓ **Downloads Two Datasets** - HAM10000 + ISIC 9-class in one command  
✓ **Auto-Unzip** - No manual extraction needed  
✓ **Class Organization** - Automatically organizes images by disease type  
✓ **Dataset Balancing** - Caps images per class for consistent training  
✓ **Statistics Reporting** - Shows final dataset composition  
✓ **No Hardcoded API Keys** - Secure credential handling  

## 📁 Output

Creates `isic_data/` directory with balanced dataset:

```
isic_data/
├── melanoma/
├── nevus/
├── basal_cell_carcinoma/
├── actinic_keratosis/
├── benign_keratosis/
├── dermatofibroma/
├── vascular_lesion/
└── ... (15-22 total classes)
```

Ready for `train.py` immediately.

## 🚀 Quick Start

### Option 1: Environment Variable (RECOMMENDED)

```bash
# Set environment variable with your API key
export KAGGLE_API_TOKEN=YOUR_API_TOKEN_HERE
export KAGGLE_USERNAME=your_kaggle_username

# Run the script
python dataset_builder.py
```

On Windows PowerShell:
```powershell
$env:KAGGLE_API_TOKEN = "YOUR_API_TOKEN_HERE"
$env:KAGGLE_USERNAME = "your_kaggle_username"
python dataset_builder.py
```

### Option 2: Command Line Arguments

```bash
python dataset_builder.py --api-key YOUR_API_TOKEN_HERE --username your_kaggle_username
```

### Option 3: kaggle.json File

If you already have `~/.kaggle/kaggle.json`:
```bash
python dataset_builder.py
```

Or place `kaggle.json` in the current directory:
```bash
# Copy kaggle.json to project directory first
cp ~/.kaggle/kaggle.json .
python dataset_builder.py
```

### Step 1: Get Kaggle API Key

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Download `kaggle.json` (or copy the key shown)

### Step 2: Install Dependencies

```bash
# Install dataset builder dependencies
pip install -r dataset_requirements.txt
```

### Step 3: Run Pipeline with Your API Key

Choose any method above to provide your API credentials.

```bash
# Using environment variable (easiest):
export KAGGLE_API_TOKEN=YOUR_KEY_HERE
python dataset_builder.py
```

The script will:
- ✓ Detect `kaggle.json` in current directory
- ✓ Move it to `~/.kaggle/`
- ✓ Set correct permissions (600)
- ✓ Download HAM10000 dataset
- ✓ Download ISIC 9-class dataset
- ✓ Unzip both automatically
- ✓ Organize by disease class
- ✓ Balance dataset (cap per class)
- ✓ Remove temporary files
- ✓ Display statistics

### Step 4: Train Model

```bash
# Install main project dependencies
pip install -r requirements.txt

# Train the model (25 epochs, ~60 min)
python train.py

# Start the web server
python app.py

# Open browser to http://localhost:5000
```

## 📊 Expected Output

```
KAGGLE DATASET BUILDER - AUTOMATED PIPELINE

🔧 SETTING UP KAGGLE API
✓ Found kaggle.json in current directory
✓ .kaggle directory: /home/user/.kaggle
✓ Copied kaggle.json to ~/.kaggle/
✓ Set permissions to 600
✓ Kaggle CLI configured successfully

📥 DOWNLOADING DATASETS
✓ Downloaded and extracted HAM10000
✓ Downloaded and extracted ISIC 9-class

📂 ORGANIZING DATASETS
✓ Organized HAM10000: 5000 images
✓ Organized ISIC 9-class: 4500 images

⚖️  BALANCING DATASET
✓ Balanced dataset: removed 1234 excess images
  Target per class: 300 images

📊 DATASET STATISTICS
✓ Total Classes: 9
✓ Total Images: 2700
✓ Images per class:
    • melanoma: 300
    • nevus: 300
    • basal_cell_carcinoma: 300
    • actinic_keratosis: 300
    • benign_keratosis: 300
    • dermatofibroma: 300
    • vascular_lesion: 300
    • seborrheic_keratosis: 300
    • squamous_cell_carcinoma: 300

✅ PIPELINE COMPLETE!
✓ Dataset ready for training!
✓ Expected training time: ~50-70 minutes (25 epochs)
```

## 🔐 Security Notes

- **NO hardcoded API keys** - Script doesn't store credentials
- **Automatic cleanup** - Kaggle.json copied to `~/.kaggle/` after setup
- **Minimal permissions** - Only reads what's needed from Kaggle
- **Temporary files removed** - No leftover data in working directory

## 📥 Datasets Used

### HAM10000
- **Source**: kmader/skin-cancer-mnist-ham10000
- **Size**: ~10,000 images
- **Classes**: 7 skin disease types
- **Use**: Primary training dataset

### ISIC 9-Class
- **Source**: nodoubttome/skin-cancer9-classesisic  
- **Size**: ~5,000 images
- **Classes**: 9 skin disease types
- **Use**: Additional diversity and classes

## 🔄 Class Mapping

Both datasets are unified into consistent class names:

| Original (HAM10000) | Mapped Name |
|---|---|
| MEL | melanoma |
| NV | nevus |
| BCC | basal_cell_carcinoma |
| AK | actinic_keratosis |
| BKL | benign_keratosis |
| DF | dermatofibroma |
| VASC | vascular_lesion |

## ⚙️ Configuration

Edit `dataset_builder.py` to modify:

- **Line ~15** - Change output directory
- **Line ~17-20** - Change Kaggle datasets
- **Line ~234** - Adjust target images per class (default: 300)

## 🐛 Troubleshooting

### "No API credentials found"

Provide your API key using ONE of these methods:

**Option 1 - Environment Variable (Best):**
```bash
# Linux/Mac
export KAGGLE_API_TOKEN=YOUR_API_TOKEN_HERE
export KAGGLE_USERNAME=your_kaggle_username

# Windows PowerShell
$env:KAGGLE_API_TOKEN = "YOUR_API_TOKEN_HERE"
$env:KAGGLE_USERNAME = "your_kaggle_username"
```

**Option 2 - Command Line:**
```bash
python dataset_builder.py --api-key YOUR_API_TOKEN_HERE --username your_username
```

**Option 3 - kaggle.json File:**
```bash
# Place kaggle.json in current directory
cp ~/.kaggle/kaggle.json .
python dataset_builder.py
```

### "kaggle.json not found in current directory!"

→ Use Option 1 (environment variable) or Option 2 (command line args)
→ Script now auto-creates kaggle.json from your API key

### "Kaggle CLI not installed"
→ Install: `pip install kaggle`

### "Kaggle CLI not found"
→ On Windows: `pip install --upgrade kaggle`
→ Restart your terminal after installing

### "Permission denied" error
→ On Windows, permission setting is skipped (safe to ignore)
→ On Linux/Mac, ensure ~/.kaggle/ has 700 permissions: `chmod 700 ~/.kaggle`

### "No internet connection"
→ Ensure you have stable internet; downloads are large (~2-3GB)
→ Kaggle may rate-limit; wait a few minutes and retry

### "Out of disk space"
→ HAM10000 + ISIC = ~2-3GB before balancing
→ Lower target_per_class in line 234 (try 200)

## 📈 Training Timeline

After running this script:

```
✓ Dataset prepared (5-10 minutes)
  ↓
✓ Run train.py (50-70 minutes)
  ↓
✓ Accuracy: 85-90%+
  ↓
✓ Run app.py (instant)
  ↓
✓ Open browser (http://localhost:5000)
```

## 📝 Files

- **dataset_builder.py** - Main automated script
- **dataset_requirements.txt** - Dependencies (just kaggle + tqdm)
- **README.md** - This file

## 💡 Notes

- Script is **fully automated** - no manual intervention needed
- Handles **interruptions gracefully** - can rerun to retry failed steps
- **Optimized for speed** - caps classes at 300 images for ~60-min total training
- **Production-ready** - proper error handling and validation

## 🎯 Next Steps

```bash
# After dataset is ready:
python train.py          # Train model (25 epochs)
python app.py            # Start web server
# Open http://localhost:5000 in browser
```

---

For issues, check the output log for specifics. The script provides detailed progress at each step.
