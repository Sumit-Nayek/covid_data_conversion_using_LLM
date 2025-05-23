# ![Medical Textifier Logo](assets/logo.png)

[![Streamlit](https://img.shields.io/badge/Streamlit-0.87-orange)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Architecture](#-architecture)
4. [Demo](#-demo)
5. [Installation](#-installation)
6. [Usage](#-usage)
7. [Project Structure](#-project-structure)
8. [Customization](#-customization)
9. [Contributing](#-contributing)
10. [License](#-license)

---

## 📌 Overview

**Medical Textifier** is a Streamlit-based web application that transforms your tabular medical data into structured, human-readable clinical summaries. Ideal for NLP preprocessing, this tool helps healthcare professionals convert CSV datasets of patient information (age, symptoms, comorbidities, RT-PCR values, and diagnosis) into coherent narrative sentences.

<p align="center">
  <img src="assets/architecture_diagram.png" alt="Architecture Diagram" width="80%">
</p>

---

## 🌟 Features

* **Easy Upload:** Drag-and-drop CSV file upload via a clean UI.
* **Dynamic Conversion:** Automatically generate summaries for each patient row.
* **Downloadable Output:** Export results as a CSV with the generated text.
* **Lightweight & Fast:** Built with Streamlit and Pandas for responsive performance.
* **Open Source:** MIT licensed; extensible for additional columns or custom logic.

---

## 🏗 Architecture

1. **Streamlit Frontend**: Handles file upload, user interactions, and result display.
2. **Conversion Logic** (`format_utils.py`): Implements sentence-mapping functions to parse and format each data row.
3. **DataFrame Processing**: Uses Pandas to read, transform, and export data.

---

## 📸 Demo

<p align="center">
  <img src="assets/screenshot.png" alt="App Screenshot" width="70%">
</p>

1. Upload your CSV file.
2. View generated clinical summaries.
3. Download the annotated dataset.

---

## ⚙️ Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/medical-textifier-app.git
   cd medical-textifier-app
   ```
2. **Create & Activate a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. **Run Locally**

   ```bash
   streamlit run app.py
   ```

2. **Interact**

   * Open the local URL (usually `http://localhost:8501`).
   * Upload your CSV, wait for conversion, and download the results.

3. **Deploy on Hugging Face Spaces**

   * Push the repo to GitHub.
   * Create a new Space on Hugging Face (SDK: Streamlit).
   * Link or upload your repository; it auto-deploys `app.py`.

---

## 🗂 Project Structure

```
medical-textifier-app/
├── app.py              # Main Streamlit app
├── format_utils.py     # Data conversion functions
├── requirements.txt    # Python dependencies
├── example_input.csv   # Sample CSV for testing
├── assets/             # Images & icons
│   ├── logo.png
│   ├── screenshot.png
│   └── architecture_diagram.png
├── README.md           # Project documentation
└── LICENSE
```

---

## 🎨 Customization

* **Add New Columns:** Update `SYMPTOM_COLS` in `format_utils.py` to include more binary symptom flags.
* **Modify Template:** Tweak `convert_row_to_text()` to adjust sentence structure or include additional metrics.
* **Styling:** Use Streamlit components (e.g., `st.sidebar`, `st.beta_columns`) for advanced UI.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Built with ❤️ by Sumit*
