# **AI-Generated Art Style Transfer Detection**
------------------------------------------------------------------------------------------------------------------------------------------------------------
**Overview**
------------------------------------------------------------------------------------------------------------------------------------------------------------
The increasing use of artificial intelligence (AI) for generating artworks through style transfer has raised several ethical concerns. While this technique allows for the creation of art by mimicking the styles of famous artists, it also threatens the integrity and ownership of original works. This project focuses on developing a reliable detection model to identify AI-generated artworks, particularly those created through style transfer techniques. By using machine learning algorithms such as ResNet50 and Vision Transformer (ViT), we aim to provide a robust solution for distinguishing between original and AI-generated art.

**Problem Statement**
------------------------------------------------------------------------------------------------------------------------------------------------------------
AI-generated art through style transfer has led to significant debates regarding copyright, artist rights, and the authenticity of creative work. Artists claim that AI-generated pieces, especially those based on their unique style, constitute plagiarism and threaten their livelihood. Therefore, detecting AI-generated artworks is essential in preserving the value of human creativity and ensuring fair attribution.

**Solution**
------------------------------------------------------------------------------------------------------------------------------------------------------------
This research proposes a solution to detect AI-generated style transfer artworks using two powerful deep learning models:

ResNet50: A Convolutional Neural Network (CNN) based on residual learning, known for its effectiveness in image recognition tasks.

Vision Transformer (ViT): A transformer-based model that divides images into patches and uses self-attention mechanisms to capture global dependencies, making it effective in identifying inconsistencies in AI-generated images.

The model is trained and evaluated on a publicly available dataset, employing techniques such as recall, precision, and F1-score to assess performance.

**Key Features**
------------------------------------------------------------------------------------------------------------------------------------------------------------
ResNet50 and ViT Models: Utilizes state-of-the-art deep learning architectures for image classification and detection.

**Data Preprocessing**
------------------------------------------------------------------------------------------------------------------------------------------------------------
Includes resizing and normalization of images to ensure compatibility with the models.

**Evaluation Metrics**
------------------------------------------------------------------------------------------------------------------------------------------------------------
The models are evaluated based on precision, recall, and F1-score, prioritizing accuracy in identifying AI-generated images.
