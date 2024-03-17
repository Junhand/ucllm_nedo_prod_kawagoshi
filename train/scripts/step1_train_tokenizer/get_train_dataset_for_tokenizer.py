import json
import os
import glob
import argparse
import spacy

def __readlines(input_file: str):
    with open(input_file) as fp:
        return fp.readlines()
    
def merge_datasets(input_dir: str, output_base: str):
    os.makedirs(output_base, exist_ok=True)
    # 入力フォルダと出力フォルダのパスを取得します。
    file_lines = {input_file: __readlines(os.path.join(input_dir, input_file))
                  for input_file in os.listdir(input_dir) if input_file.endswith("results.dedup.jsonl")}
    # 入力フォルダ内の全てのJSONLファイルを処理します。
    for input_file, json_lines in file_lines.items():
        texts = []  # テキストを格納するためのリストを初期化します。
        
        for line in json_lines:
            data = json.loads(line)
            nlp = spacy.load('ja_ginza')
            try:
                doc = nlp(data['text']) # テキストからSentenceを抽出します。
                for sent in doc.sents:
                    texts.append(sent.text)
            except:
                continue
        all_texts = '\n'.join(texts)  # テキストを連結します。
        
        # 出力ファイルのパスを生成します。
        output_file_name = os.path.basename(input_file).replace('.jsonl', '.txt')
        output_file_path = os.path.join(output_base, output_file_name)
        
        # 結果をTXTファイルに保存します。
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(all_texts)
        
        print(f"{input_dir} からテキストを抽出し、{output_file_path} に保存しました。")

def main():
    parser = argparse.ArgumentParser(description='JSONLファイルからテキストを抽出し、TXTファイルとして保存します。')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=False, default="/home/ubuntu/ucllm_nedo_prod_kawagoshi/data_management/output/debuped_documents")
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="/home/ubuntu/ucllm_nedo_prod_kawagoshi/train/scripts/step1_train_tokenizer/dataset")
    args = parser.parse_args()

    #start = datetime.now()
    #output_base = os.path.join(args.output_dir, start.strftime("%Y%m%d%H%M%S"))
    output_base = os.path.join(args.output_dir, "original_txt")

    merge_datasets(input_dir=args.input_dir, output_base=output_base)

if __name__ == "__main__":
    main()