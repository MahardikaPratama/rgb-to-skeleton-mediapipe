import pandas as pd
import os
import json

def update_notebook_and_generate_txt():
    base_dir = r'c:\TA\Source-Code\rgb-to-skeleton-mediapipe\splitting_data'
    excel_path = os.path.join(base_dir, 'Gloss dan Tanda Dataset.xlsx')
    
    # 1. Generate TXT files from existing CSVs and Excel
    try:
        df_excel = pd.read_excel(excel_path)
        mapping_text = {}
        for _, row in df_excel.dropna(subset=['ID']).iterrows():
            mapping_text[str(row['ID']).strip()] = str(row['Kalimat']).strip()
            
        for split in ['train', 'dev', 'test']:
            csv_path = os.path.join(base_dir, f'{split}.csv')
            txt_path = os.path.join(base_dir, f'sd_{split}_list.txt')
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, sep='|')
                # Extract sentence ID from Pxx_Sxxx_Rxx format
                df['text'] = df['id'].apply(lambda x: mapping_text.get(x.split('_')[1] if len(x.split('_')) >= 2 else '', f"UNKNOWN_{x}"))
                df[['id', 'gloss', 'text']].to_csv(txt_path, sep='|', index=False)
                print(f'✅ Berhasil membuat {os.path.basename(txt_path)}')
            else:
                print(f'❌ File {os.path.basename(csv_path)} tidak ditemukan, pastikan sudah menjalankan notebook sebelumnya.')
    except Exception as e:
        print(f"Error saat membuat file txt: {e}")

    # 2. Update data_splitting.ipynb
    try:
        ipynb_path = os.path.join(base_dir, 'data_splitting.ipynb')
        if os.path.exists(ipynb_path):
            with open(ipynb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            updated = False
            for cell in nb.get('cells', []):
                if cell['cell_type'] == 'code':
                    source = "".join(cell['source'])
                    
                    # Update mapping logic
                    if "col_sentence = 'ID'" in source and "col_text = 'Kalimat'" not in source:
                        new_source = source.replace("col_gloss = 'Gloss'      # Ubah sesuai header di Excel", 
                                                    "col_gloss = 'Gloss'      # Ubah sesuai header di Excel\ncol_text = 'Kalimat'     # Tambahan untuk teks asli")
                        new_source = new_source.replace("mapping = {}", "mapping = {}\nmapping_text = {}")
                        new_source = new_source.replace("mapping[str(row[col_sentence]).strip()] = str(row[col_gloss]).strip()", 
                                                        "mapping[str(row[col_sentence]).strip()] = str(row[col_gloss]).strip()\n        if col_text in df_excel.columns:\n            mapping_text[str(row[col_sentence]).strip()] = str(row[col_text]).strip()")
                        cell['source'] = [line + ('\n' if i < len(new_source.split('\n'))-1 else '') for i, line in enumerate(new_source.split('\n'))]
                        updated = True
                        
                    # Update dataframe building
                    if "gloss = mapping.get(sentence, f\"UNKNOWN_{sentence}\")" in source and "text = mapping_text" not in source:
                        new_source = source.replace("gloss = mapping.get(sentence, f\"UNKNOWN_{sentence}\")",
                                                    "gloss = mapping.get(sentence, f\"UNKNOWN_{sentence}\")\n        text = mapping_text.get(sentence, f\"UNKNOWN_{sentence}\")")
                        new_source = new_source.replace("'gloss': gloss,", "'gloss': gloss,\n            'text': text,")
                        cell['source'] = [line + ('\n' if i < len(new_source.split('\n'))-1 else '') for i, line in enumerate(new_source.split('\n'))]
                        updated = True

                    # Update CSV output
                    if "df_train[output_columns].to_csv('train.csv'" in source and "sd_train_list.txt" not in source:
                        new_source = source.replace("output_columns = ['id', 'gloss']",
                                                    "output_columns = ['id', 'gloss']\noutput_columns_txt = ['id', 'gloss', 'text']")
                        new_source = new_source.replace("df_test[output_columns].to_csv('test.csv', index=False, sep='|')",
                                                        "df_test[output_columns].to_csv('test.csv', index=False, sep='|')\n\noutput_columns_txt = ['id', 'gloss', 'text']\ndf_train[output_columns_txt].to_csv('sd_train_list.txt', index=False, sep='|')\ndf_dev[output_columns_txt].to_csv('sd_dev_list.txt', index=False, sep='|')\ndf_test[output_columns_txt].to_csv('sd_test_list.txt', index=False, sep='|')")
                        new_source = new_source.replace("print(\"Selesai! File train.csv, dev.csv, dan test.csv siap digunakan.\")",
                                                        "print(\"Selesai! File train.csv, dev.csv, test.csv, serta file list TXT siap digunakan.\")")
                        cell['source'] = [line + ('\n' if i < len(new_source.split('\n'))-1 else '') for i, line in enumerate(new_source.split('\n'))]
                        updated = True
                        
            if updated:
                with open(ipynb_path, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=1)
                print("✅ Berhasil mengupdate data_splitting.ipynb")
            else:
                print("ℹ️ Notebook data_splitting.ipynb sepertinya sudah diupdate sebelumnya.")
        else:
            print("❌ Notebook data_splitting.ipynb tidak ditemukan.")
    except Exception as e:
        print(f"Error saat mengupdate notebook: {e}")

if __name__ == "__main__":
    update_notebook_and_generate_txt()
