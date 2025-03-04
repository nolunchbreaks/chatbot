# 🛠️ Recovery Chatbot 🤖

A **supportive AI chatbot** designed to help individuals in recovery by offering encouragement, guidance, and resources. Built using **Hugging Face's Transformers** and fine-tuned on recovery-related conversations.

---

## 🚀 Features

- 🤖 AI-powered conversations for recovery support
- 📚 Trained with **BlenderBot 400M** for natural responses
- 🎯 Fine-tuned on **custom recovery data**
- ⚡ Uses **PyTorch & Hugging Face Transformers**
- 🔗 Open for **community contributions**

---

## 📦 Installation

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/nolunchbreaks/chatbot.git
cd chatbot
```

### 2️⃣ Set Up Virtual Environment

```sh
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 4️⃣ Run the Chatbot

```sh
python chatbot.py
```

---

## 🏋️‍♂️ Training the Chatbot

### 1️⃣ Prepare Data

Add recovery conversations to `recovery_data.txt`

### 2️⃣ Run Training:

```sh
python tune_bot.py
```

### 3️⃣ Save the Model:

```sh
python save_model.py
```

---

## 🤝 Contributing

We welcome contributions! To get started:

### 1️⃣ Fork the repository

### 2️⃣ Create a feature branch:

```sh
git checkout -b feature-branch
```

### 3️⃣ Make your changes & commit:

```sh
git commit -m "Added a new feature"
```

### 4️⃣ Push & submit a pull request:

```sh
git push origin feature-branch
```

---

## 📝 License

This project is open-source under the MIT License.

---

💙 "Recovery is a journey, not a destination."

