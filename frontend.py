import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
from VotingEnsemble import VotingEnsemble

# Class names and info
CLASS_NAMES = {
    0: 'Diabetic_Retinopathy',
    1: 'Normal',
    2: 'Cataract',
    3: 'Glaucoma'
}

CLASS_INFO = {
    'Diabetic_Retinopathy': 'Include a variety of nutrient-rich foods: fruits, vegetables, whole grains, lean proteins, and healthy fats. Consume sources like fatty fish, flaxseeds, chia seeds, and walnuts for potential anti-inflammatory effects. Eat foods high in antioxidants, such as berries, citrus fruits, dark leafy greens, carrots, and nuts. Reduce saturated and trans fats found in fried and processed foods; opt for healthier fats like those in olive oil and nuts. Keep levels in check by limiting salt, choosing heart-healthy foods, and following medical advice.',
    'Normal': 'No visible retinal abnormalities. Stay Healthy !!! Stay Safe !!!',
    'Cataract': 'Eat foods high in antioxidants, including fruits, vegetables, and nuts. Include sources like fatty fish, flaxseeds, and walnuts. Ensure an adequate intake of vitamins A, C, and E, as well as zinc and selenium. Reduce intake of processed foods and excessive sugar. Consume lutein and zeaxanthin found in leafy greens and colorful vegetables.',
    'Glaucoma': 'Emphasize fruits, vegetables, whole grains, lean proteins, and healthy fats. Reduce caffeine intake; moderate sodium to manage intraocular pressure. Include sources like fatty fish, flaxseeds, and walnuts.'
}

# Load model
checkpoint = torch.load('./retinal.pth', map_location=torch.device('cpu'))
vision_transformer = timm.create_model('vit_base_patch16_224', num_classes=2)
vision_transformer.load_state_dict(checkpoint['vision_transformer_state_dict'])

resnet101 = timm.create_model('resnet101', num_classes=2)
filtered_state_dict = {k: v for k, v in checkpoint['resnet101_state_dict'].items() if k in resnet101.state_dict()}
resnet101.load_state_dict(filtered_state_dict, strict=False)

ensemble_model = VotingEnsemble([vision_transformer, resnet101])
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ensemble_model = ensemble_model.to(device)
ensemble_model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_image(img):
    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        outputs = ensemble_model(img)
        _, predicted = outputs.max(1)

    return predicted.item()


def main():
    st.title("Retinal Disease Classification")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the selected image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Transform and make prediction
        img = transform(image)
        predicted_class = predict_image(img)

        # Print results
        st.subheader('Prediction:')
        st.write('Predicted class:', CLASS_NAMES[predicted_class])
        st.write('Class information:', CLASS_INFO[CLASS_NAMES[predicted_class]])


if __name__ == '__main__':
    main()
