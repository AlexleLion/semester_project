import os
import pytesseract
import os
import pandas as pd
import cv2

from PIL import Image
from pdf2image import convert_from_path

def rename_pdfs(dossier_pdf):
    # Liste des fichiers PDF dans le dossier
    pdf_files = [f for f in os.listdir(dossier_pdf) if f.endswith('.pdf')]

    # Renommer les fichiers en ajoutant un numéro en ordre croissant
    for i, filename in enumerate(sorted(pdf_files), start=1):
        new_filename = f"{i:02d}_{filename}"
        os.rename(os.path.join(dossier_pdf, filename), os.path.join(dossier_pdf, new_filename))

    # Afficher les nouveaux noms de fichiers
    print("Fichiers renommés :")
    for filename in sorted(os.listdir(dossier_pdf)):
        if filename.endswith('.pdf'):
            print(filename)
            
def convert_pdfs_to_images(dossier_pdf):

    # Liste des fichiers PDF dans le dossier
    pdf_files = [f for f in os.listdir(dossier_pdf) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        # Transformer chaque PDF en fichiers JPG
        # Créer un dossier pour chaque PDF
        pdf_name = os.path.splitext(pdf_file)[0]

        output_folder = os.path.join(dossier_pdf, pdf_name[:9])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        # Chemin complet du fichier PDF
        pdf_path = os.path.join(dossier_pdf, pdf_file)
        print(pdf_path)
        
        # Convertir le PDF en images
        images = convert_from_path(pdf_path)

        # Sauvegarder chaque page en tant que fichier JPG
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f'page_{i+1}.jpg')
            image.save(image_path, 'JPEG')

    print("Conversion terminée.")
        

def convert_images_to_text(dossier_images):
    dossier_jpgs = [f.path for f in os.scandir(dossier_images) if f.is_dir()]

    # Liste des fichiers JPG dans le dossier
    for dossier_jpg in dossier_jpgs:
        jpg_files = [f for f in os.listdir(dossier_jpg) if f.endswith('.jpg')]

        # Extraire le texte de chaque image et stocker dans une liste
        data = []
        for jpg_file in jpg_files:
            image_path = os.path.join(dossier_jpg, jpg_file)
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.fastNlMeansDenoising(image, None, 8, 10, 21)

            final_image = Image.fromarray(image)

            text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
            data.append({'filename': jpg_file, 'text': text})

            # Convertir la liste en DataFrame
            df = pd.DataFrame(data)

            # Sauvegarder en TXT
            dossier_nom = os.path.basename(dossier_jpg)
            
            txt_dir = f'../data/txt/ouvrages_feminins/{dossier_nom}'

            if not os.path.exists(txt_dir):
                os.makedirs(txt_dir, exist_ok=True)
                
            txt_path = os.path.join(txt_dir, os.path.splitext(jpg_file)[0] + '.txt')
            print(txt_path)
            with open(txt_path, 'w', encoding='utf-8') as file:
                file.write(text)
                
    print("Océrisation terminée.")