import os
import io
import re
import fitz
import docx
import csv
import easyocr
import numpy as np
from PIL import Image
from pptx import Presentation
from openpyxl import load_workbook
import tempfile
import json
import uuid
from flask import Flask, request, jsonify, render_template, send_file


class DocumentParser:

    def __init__(self, language_list=None, chunk_size=1000, overlap=True, overlap_limit=200):

        if language_list is None:
            language_list = ['ch_sim', 'en']  # 默认支持中文和英文
        self.reader = easyocr.Reader(language_list)  # 初始化EasyOCR
        self.languages = language_list
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.overlap_limit = overlap_limit  # 新增：重叠字数上限

    def parse(self, file_path, chunk_size=None):
        if chunk_size is not None:
            self.chunk_size = chunk_size

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == '.pdf':
            return self.parse_pdf(file_path)
        elif ext == '.docx':
            return self.parse_docx(file_path)
        elif ext == '.txt':
            return self.parse_txt(file_path)
        elif ext == '.pptx':
            return self.parse_pptx(file_path)
        elif ext in ['.xlsx', '.xls']:
            return self.parse_excel(file_path)
        elif ext in ['.csv']:
            return self.parse_csv(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def parse_pdf(self, file_path):
        doc = fitz.open(file_path)
        text_segments = []
        last_sentence = None

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            page_text = page.get_text()

            if len(page_text.strip()) < 50:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                img_np = np.array(img)

                ocr_results = self.reader.readtext(img_np)
                page_text_from_ocr = ' '.join([text for _, text, _ in ocr_results])

                if page_text_from_ocr.strip():
                    page_text = page_text_from_ocr

            # 应用重叠逻辑
            page_text = self._apply_overlap(page_text, last_sentence)

            page_segments = self.segment_text(page_text, self.chunk_size)

            if page_segments:
                last_segment = page_segments[-1]
                last_sentences = self.split_into_sentences(last_segment)
                if last_sentences:
                    last_sentence = last_sentences[-1]

            for segment in page_segments:
                text_segments.append({
                    'content': segment,
                    'page': page_num + 1,
                    'type': 'text'
                })

            image_list = page.get_images(full=True)
            last_image_sentence = None

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    img_np = np.array(img)

                    ocr_results = self.reader.readtext(img_np)
                    image_text = ' '.join([text for _, text, _ in ocr_results])

                    if image_text.strip():
                        # 应用重叠逻辑
                        image_text = self._apply_overlap(image_text, last_image_sentence or last_sentence)

                        ocr_segments = self.segment_text(image_text, self.chunk_size)

                        if ocr_segments:
                            last_segment = ocr_segments[-1]
                            last_sentences = self.split_into_sentences(last_segment)
                            if last_sentences:
                                last_image_sentence = last_sentences[-1]
                                last_sentence = last_image_sentence

                        for segment in ocr_segments:
                            text_segments.append({
                                'content': segment,
                                'page': page_num + 1,
                                'type': 'image_text',
                                'image_index': img_index
                            })
                except Exception as e:
                    print(f"处理图片时出错: {e}")

        return text_segments

    def parse_docx(self, file_path):
        doc = docx.Document(file_path)
        text_segments = []
        last_sentence = None

        paragraphs_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs_text.append(para.text)

        if paragraphs_text:
            combined_text = "\n".join(paragraphs_text)
            segments = self.segment_text(combined_text, self.chunk_size)

            if segments:
                last_segment = segments[-1]
                last_sentences = self.split_into_sentences(last_segment)
                if last_sentences:
                    last_sentence = last_sentences[-1]

            for segment in segments:
                text_segments.append({
                    'content': segment,
                    'type': 'text'
                })

        for table_index, table in enumerate(doc.tables):
            table_text = []
            for row in table.rows:
                row_text = [cell.text for cell in row.cells]
                table_text.append(' | '.join(row_text))

            if table_text:
                combined_text = '\n'.join(table_text)

                # 应用重叠逻辑
                combined_text = self._apply_overlap(combined_text, last_sentence)

                table_segments = self.segment_text(combined_text, self.chunk_size)

                if table_segments:
                    last_segment = table_segments[-1]
                    last_sentences = self.split_into_sentences(last_segment)
                    if last_sentences:
                        last_sentence = last_sentences[-1]

                for segment in table_segments:
                    text_segments.append({
                        'content': segment,
                        'table': table_index + 1,
                        'type': 'table'
                    })

        return text_segments

    def parse_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()

        segments = self.segment_text(text, self.chunk_size)
        return [{'content': segment, 'type': 'text'} for segment in segments if segment.strip()]

    def parse_pptx(self, file_path):
        prs = Presentation(file_path)
        text_segments = []
        last_sentence = None

        for slide_index, slide in enumerate(prs.slides):
            slide_text = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text)

            if slide_text:
                combined_text = '\n'.join(slide_text)

                # 应用重叠逻辑，确保即使在空白幻灯片后也能正确处理
                combined_text = self._apply_overlap(combined_text, last_sentence)

                slide_segments = self.segment_text(combined_text, self.chunk_size)

                if slide_segments:
                    # 更新last_sentence为本张幻灯片最后一段的最后一句
                    last_segment = slide_segments[-1]
                    last_sentences = self.split_into_sentences(last_segment)
                    if last_sentences:
                        last_sentence = last_sentences[-1]

                for segment in slide_segments:
                    text_segments.append({
                        'content': segment,
                        'slide': slide_index + 1,
                        'type': 'slide'
                    })

        return text_segments

    def parse_excel(self, file_path):
        """解析Excel文件"""
        wb = load_workbook(file_path, data_only=True)
        text_segments = []
        last_sentence = None  # 用于跨表格的重叠

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            sheet_data = []

            for row in sheet.iter_rows(values_only=True):
                if any(cell for cell in row):  # 跳过空行
                    row_text = ' | '.join(str(cell) for cell in row if cell is not None)
                    sheet_data.append(row_text)

            if sheet_data:
                combined_text = '\n'.join(sheet_data)

                # 应用重叠逻辑
                combined_text = self._apply_overlap(combined_text, last_sentence)

                sheet_segments = self.segment_text(combined_text, self.chunk_size)

                if sheet_segments:
                    # 更新last_sentence为本表格最后一段的最后一句
                    last_segment = sheet_segments[-1]
                    last_sentences = self.split_into_sentences(last_segment)
                    if last_sentences:
                        last_sentence = last_sentences[-1]

                for segment in sheet_segments:
                    text_segments.append({
                        'content': segment,
                        'sheet': sheet_name,
                        'type': 'spreadsheet'
                    })

        return text_segments

    def parse_csv(self, file_path):
        """解析CSV文件"""
        text_segments = []
        rows = []
        last_sentence = None  # 添加用于CSV的last_sentence追踪

        with open(file_path, 'r', newline='', encoding='utf-8', errors='ignore') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if any(cell.strip() for cell in row):  # 跳过空行
                    rows.append(' | '.join(row))

        if rows:
            combined_text = '\n'.join(rows)

            # 应用重叠逻辑 (新增)
            combined_text = self._apply_overlap(combined_text, last_sentence)

            csv_segments = self.segment_text(combined_text, self.chunk_size)

            if csv_segments:
                # 更新last_sentence (新增)
                last_segment = csv_segments[-1]
                last_sentences = self.split_into_sentences(last_segment)
                if last_sentences:
                    last_sentence = last_sentences[-1]

            for segment in csv_segments:
                text_segments.append({
                    'content': segment,
                    'type': 'csv'
                })

        return text_segments

    def _apply_overlap(self, current_text, last_sentence):
        """
        应用重叠逻辑的统一方法，确保各文件类型一致处理重叠

        Args:
            current_text: 当前要处理的文本
            last_sentence: 上一段落/文件的最后一句话

        Returns:
            应用重叠后的文本
        """
        if not current_text.strip() or not self.overlap or last_sentence is None:
            return current_text

        # 确保current_text不是空的
        current_text = current_text.strip()
        last_sentence = last_sentence.strip()

        # 检查current_text是否已经以last_sentence开始
        if current_text.startswith(last_sentence):
            return current_text

        # 如果last_sentence太长，截取到不超过overlap_limit的长度
        if len(last_sentence) > self.overlap_limit:
            # 尝试在单词边界截断
            words = last_sentence.split()
            truncated = ""
            for word in words:
                if len(truncated + word) < self.overlap_limit:
                    truncated += word + " "
                else:
                    break
            last_sentence = truncated.strip()

            # 如果没有成功在单词边界截断，直接截取
            if len(last_sentence) > self.overlap_limit:
                last_sentence = last_sentence[:self.overlap_limit]

        # 应用重叠
        return last_sentence + " " + current_text

    def split_into_sentences(self, text):
        """
        将文本分割成句子 - 支持中英文混合文本

        Args:
            text: 要分割的文本

        Returns:
            句子列表
        """
        if not text or not text.strip():
            return []

        # 预处理：保留换行符，但清理多余空白
        text = re.sub(r'[ \t]+', ' ', text).strip()

        # 合并不必要的多行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 句子结束标记模式（支持中英文标点）
        sent_end_pattern = r'([.。!！?？;；]\s*)'

        # 使用正则表达式分割句子
        sentences = []
        segments = re.split(sent_end_pattern, text)

        # 重组句子（保留标点）
        i = 0
        current_sentence = ""
        while i < len(segments):
            if i + 1 < len(segments) and re.match(sent_end_pattern, segments[i + 1]):
                # 将句子与其标点合并
                current_sentence += segments[i] + segments[i + 1]
                sentences.append(current_sentence.strip())
                current_sentence = ""
                i += 2
            else:
                # 对于没有标准句尾标点的文本
                if segments[i].strip():
                    current_sentence += segments[i]
                i += 1

        # 处理最后可能剩余的片段
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # 处理段落分隔符
        result = []
        for sentence in sentences:
            # 按段落换行符分割
            paragraph_parts = re.split(r'\n\s*\n', sentence)

            for part in paragraph_parts:
                if part.strip():
                    # 检查是否有换行符但不是段落分隔符
                    if '\n' in part and not re.search(r'\n\s*\n', part):
                        # 对于长的单行，按换行符分割
                        subparts = part.split('\n')
                        for subpart in subparts:
                            if subpart.strip():
                                result.append(subpart.strip())
                    else:
                        result.append(part.strip())

        # 过滤空句子并确保每个句子以句号结尾
        final_sentences = []
        for s in result:
            if not s.strip():
                continue

            # 如果句子不以标准标点结尾，添加一个句号
            if not re.search(r'[.。!！?？;；]$', s):
                s = s + '。'

            final_sentences.append(s)

        return final_sentences

    def segment_text(self, text, max_length=1000):
        """
        将文本分割成重叠的语义段落

        Args:
            text: 要分割的文本
            max_length: 每个段落的最大长度

        Returns:
            文本段落列表
        """
        if not text or not text.strip():
            return []

        # 将文本分割成句子
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        # 创建重叠的段落
        chunks = []
        current_chunk = []
        current_length = 0
        last_sentence = None

        for sentence in sentences:
            # 如果当前句子太长，可以单独作为一个段落
            if len(sentence) > max_length:
                # 先完成当前段落
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    last_sentence = current_chunk[-1] if current_chunk else None

                # 将长句子作为单独段落
                chunks.append(sentence)
                last_sentence = sentence
                current_chunk = []
                current_length = 0

                # 如果设置了重叠，将该句子作为下一段的第一句
                if self.overlap and last_sentence:
                    # 应用重叠限制
                    overlap_sentence = last_sentence
                    if len(overlap_sentence) > self.overlap_limit:
                        overlap_sentence = overlap_sentence[:self.overlap_limit]

                    current_chunk = [overlap_sentence]
                    current_length = len(overlap_sentence) + 1  # +1 for space
                continue

            # 检查添加当前句子是否会超过最大长度
            if current_length + len(sentence) + 1 > max_length and current_chunk:
                # 保存当前段落
                chunks.append(" ".join(current_chunk))

                # 如果设置了重叠，使用当前段落的最后一句作为下一段落的第一句
                if self.overlap:
                    last_sentence = current_chunk[-1]

                    # 应用重叠限制
                    overlap_sentence = last_sentence
                    if len(overlap_sentence) > self.overlap_limit:
                        overlap_sentence = overlap_sentence[:self.overlap_limit]

                    current_chunk = [overlap_sentence]
                    current_length = len(overlap_sentence) + 1  # +1 for space
                else:
                    current_chunk = []
                    current_length = 0

            # 添加当前句子到段落
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for space

        # 添加最后一个段落
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def extract_text_from_image(self, image_path):
        """使用EasyOCR从图像中提取文本"""
        results = self.reader.readtext(image_path)
        return ' '.join([text for _, text, _ in results])

    def extract_images_from_pdf(self, pdf_path, output_dir=None):
        """从PDF中提取图像并对其执行OCR"""
        if output_dir is None:
            output_dir = tempfile.mkdtemp()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        doc = fitz.open(pdf_path)
        image_text_segments = []
        last_sentence = None  # 用于实现图片OCR文本之间的重叠

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # 将图像保存到临时文件
                temp_image_path = os.path.join(output_dir, f"page{page_num + 1}_img{img_index + 1}.png")
                with open(temp_image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                # 对图像执行OCR
                try:
                    ocr_text = self.extract_text_from_image(temp_image_path)
                    if ocr_text.strip():
                        # 应用重叠逻辑
                        ocr_text = self._apply_overlap(ocr_text, last_sentence)

                        # 对OCR结果进行分段
                        ocr_segments = self.segment_text(ocr_text, self.chunk_size)

                        if ocr_segments:
                            # 更新最后一句
                            last_segment = ocr_segments[-1]
                            last_sentences = self.split_into_sentences(last_segment)
                            if last_sentences:
                                last_sentence = last_sentences[-1]

                        for segment in ocr_segments:
                            image_text_segments.append({
                                'content': segment,
                                'page': page_num + 1,
                                'image_index': img_index + 1,
                                'type': 'image_text',
                                'image_path': temp_image_path
                            })
                except Exception as e:
                    print(f"对图像执行OCR时出错: {e}")

        return image_text_segments


# 新增: 处理多个文档的函数
def process_multiple_documents(file_paths, languages=None, chunk_size=1000, overlap=True, overlap_limit=200):
    """
    处理多个文档并返回其分段内容

    Args:
        file_paths: 文件路径列表
        languages: OCR支持的语言列表
        chunk_size: 每个文本块的最大字符数
        overlap: 是否启用重叠模式
        overlap_limit: 重叠文本的最大字符数

    Returns:
        以文件路径为键，分段内容为值的字典
    """
    if languages is None:
        languages = ['ch_sim', 'en']  # 默认支持中文和英文

    parser = DocumentParser(language_list=languages, chunk_size=chunk_size, overlap=overlap,
                            overlap_limit=overlap_limit)
    results = {}

    for file_path in file_paths:
        try:
            segments = parser.parse(file_path)
            results[file_path] = segments
        except Exception as e:
            results[file_path] = {"error": str(e)}

    return results


# 原始的单文件处理函数
def process_document(file_path, languages=None, chunk_size=1000, overlap=True, overlap_limit=200):
    """
    处理文档并返回其分段内容

    Args:
        file_path: 文件路径
        languages: OCR支持的语言列表
        chunk_size: 每个文本块的最大字符数
        overlap: 是否启用重叠模式
        overlap_limit: 重叠文本的最大字符数
    """
    if languages is None:
        languages = ['ch_sim', 'en']  # 默认支持中文和英文
    parser = DocumentParser(language_list=languages, chunk_size=chunk_size, overlap=overlap,
                            overlap_limit=overlap_limit)
    segments = parser.parse(file_path)
    return segments


# 保存分段结果到文件
def save_segments(segments, output_path):
    """将分段结果保存到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {output_path}")


# 创建Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 限制上传大小为50MB
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    """网页界面首页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """处理文件上传请求"""
    if 'files' not in request.files:
        return jsonify({"error": "请求中没有文件部分"}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "未选择文件"}), 400

    # 获取参数
    languages = request.form.get('languages', 'ch_sim,en').split(',')
    chunk_size = int(request.form.get('chunk_size', 1000))
    overlap = request.form.get('overlap', 'true').lower() == 'true'
    overlap_limit = int(request.form.get('overlap_limit', 200))  # 新增：获取重叠字数上限

    saved_files = []
    try:
        # 保存上传的文件
        for file in files:
            # 创建唯一文件名以避免冲突
            filename = str(uuid.uuid4()) + '_' + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_files.append(file_path)

        # 处理文件
        results = process_multiple_documents(
            saved_files,
            languages=languages,
            chunk_size=chunk_size,
            overlap=overlap,
            overlap_limit=overlap_limit  # 传递重叠字数上限
        )

        # 清理文件
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass

        # 转换结果以包含原始文件名
        final_results = {}
        for i, file_path in enumerate(saved_files):
            original_filename = files[i].filename
            final_results[original_filename] = results[file_path]

        return jsonify(final_results)

    except Exception as e:
        # 出错时清理文件
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass
        return jsonify({"error": str(e)}), 500


@app.route('/api/process', methods=['POST'])
def api_process():
    """API接口，无需Web界面即可处理文件"""
    if 'files' not in request.files:
        return jsonify({"error": "请求中没有文件部分"}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "未选择文件"}), 400

    # 获取参数
    languages = request.form.get('languages', 'ch_sim,en').split(',')
    chunk_size = int(request.form.get('chunk_size', 1000))
    overlap = request.form.get('overlap', 'true').lower() == 'true'
    overlap_limit = int(request.form.get('overlap_limit', 200))  # 新增：获取重叠字数上限

    saved_files = []
    try:
        # 保存上传的文件
        for file in files:
            # 创建唯一文件名以避免冲突
            filename = str(uuid.uuid4()) + '_' + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_files.append(file_path)

        # 处理文件
        results = process_multiple_documents(
            saved_files,
            languages=languages,
            chunk_size=chunk_size,
            overlap=overlap,
            overlap_limit=overlap_limit  # 传递重叠字数上限
        )

        # 清理文件
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass

        # 转换结果以包含原始文件名
        final_results = {}
        for i, file_path in enumerate(saved_files):
            original_filename = files[i].filename
            final_results[original_filename] = results[file_path]

        return jsonify(final_results)

    except Exception as e:
        # 出错时清理文件
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass
        return jsonify({"error": str(e)}), 500


@app.route('/api/download_results', methods=['POST'])
def download_results():
    """将处理结果作为JSON文件下载"""
    if not request.json:
        return jsonify({"error": "没有提供JSON数据"}), 400

    try:
        # 创建临时文件来存储结果
        fd, temp_file_path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(request.json, f, ensure_ascii=False, indent=2)

        # 发送文件
        return send_file(
            temp_file_path,
            as_attachment=True,
            download_name="document_parser_results.json",
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({"error": f"创建下载文件时出错: {str(e)}"}), 500


# 命令行使用的主函数
def main():
    import argparse

    parser = argparse.ArgumentParser(description='解析文档并提取文本，支持OCR')

    # 主要子命令
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 处理单个文件的子命令
    parser_single = subparsers.add_parser('process', help='处理单个文件')
    parser_single.add_argument('file_path', help='要解析的文档路径')
    parser_single.add_argument('--languages', nargs='+', default=['ch_sim', 'en'],
                               help='OCR支持的语言 (例如: ch_sim en ja)')
    parser_single.add_argument('--output', help='结果输出文件路径（JSON格式）')
    parser_single.add_argument('--chunk_size', type=int, default=1000,
                               help='每个段落的最大字符数')
    parser_single.add_argument('--no-overlap', action='store_true',
                               help='禁用段落重叠模式（默认启用重叠）')
    parser_single.add_argument('--overlap_limit', type=int, default=200,
                               help='重叠文本的最大字符数')

    # 处理多个文件的子命令
    parser_multi = subparsers.add_parser('process_multiple', help='处理多个文件')
    parser_multi.add_argument('file_paths', nargs='+', help='要解析的多个文档路径')
    parser_multi.add_argument('--languages', nargs='+', default=['ch_sim', 'en'],
                              help='OCR支持的语言 (例如: ch_sim en ja)')
    parser_multi.add_argument('--output', help='结果输出文件路径（JSON格式）')
    parser_multi.add_argument('--chunk_size', type=int, default=1000,
                              help='每个段落的最大字符数')
    parser_multi.add_argument('--no-overlap', action='store_true',
                              help='禁用段落重叠模式（默认启用重叠）')
    parser_multi.add_argument('--overlap_limit', type=int, default=200,
                              help='重叠文本的最大字符数')

    # 启动Web服务器的子命令
    parser_server = subparsers.add_parser('server', help='启动Web服务器')
    parser_server.add_argument('--host', default='0.0.0.0', help='服务器主机地址')
    parser_server.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser_server.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    if args.command is None and hasattr(args, 'file_path'):
        args.command = 'process'

    try:
        if args.command == 'process':
            parser = DocumentParser(
                language_list=args.languages,
                chunk_size=args.chunk_size,
                overlap=not args.no_overlap,
                overlap_limit=args.overlap_limit
            )

            results = parser.parse(args.file_path)

            for i, segment in enumerate(results, 1):
                print(f"段落 {i}:")
                print(f"类型: {segment.get('type', 'text')}")
                if 'page' in segment:
                    print(f"页码: {segment['page']}")
                print(f"内容: {segment['content'][:100]}..." if len(
                    segment['content']) > 100 else f"内容: {segment['content']}")
                print("-" * 80)

            if args.output:
                save_segments(results, args.output)

        elif args.command == 'process_multiple':
            results = process_multiple_documents(
                args.file_paths,
                languages=args.languages,
                chunk_size=args.chunk_size,
                overlap=not args.no_overlap,
                overlap_limit=args.overlap_limit
            )

            for file_path, segments in results.items():
                print(f"文件: {file_path}")
                print(f"段落数量: {len(segments)}")
                print("-" * 80)

            if args.output:
                save_segments(results, args.output)

        elif args.command == 'server':
            template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
            if not os.path.exists(template_dir):
                os.makedirs(template_dir)

            with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
                f.write('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>文档解析器</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                        .form-group { margin-bottom: 15px; }
                        label { display: block; margin-bottom: 5px; }
                        button { padding: 10px 15px; background: #4CAF50; color: white; border: none; cursor: pointer; }
                        button:hover { background: #45a049; }
                        .results { margin-top: 20px; border: 1px solid #ddd; padding: 15px; display: none; }
                        pre { background: #f5f5f5; padding: 10px; overflow-x: auto; }
                        .file-list { margin-bottom: 10px; }
                        .file-item { background: #f9f9f9; padding: 5px; margin: 5px 0; border-radius: 3px; }
                        .loading { display: none; margin-top: 20px; }
                        .spinner { 
                            border: 4px solid #f3f3f3;
                            border-top: 4px solid #3498db;
                            border-radius: 50%;
                            width: 30px;
                            height: 30px;
                            animation: spin 2s linear infinite;
                            display: inline-block;
                            vertical-align: middle;
                            margin-right: 10px;
                        }
                        @keyframes spin {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(360deg); }
                        }
                        .download-btn { 
                            background: #2196F3; 
                            margin-top: 10px; 
                            display: none; 
                        }
                    </style>
                </head>
                <body>
                    <h1>文档解析器</h1>
                    <form id="uploadForm" enctype="multipart/form-data">
                        <div class="form-group">
                            <label for="files">选择文件 (支持多文件):</label>
                            <input type="file" id="files" name="files" multiple required>
                        </div>
                        <div id="fileList" class="file-list"></div>
                        <div class="form-group">
                            <label for="languages">语言 (逗号分隔):</label>
                            <input type="text" id="languages" name="languages" value="ch_sim,en">
                        </div>
                        <div class="form-group">
                            <label for="chunk_size">最大段落:</label>
                            <input type="number" id="chunk_size" name="chunk_size" value="1000">
                        </div>
                        <div class="form-group">
                            <label for="overlap">启用重叠:</label>
                            <input type="checkbox" id="overlap" name="overlap" checked>
                        </div>
                        <div class="form-group">
                            <label for="overlap_limit">重叠字数上限:</label>
                            <input type="number" id="overlap_limit" name="overlap_limit" value="200">
                        </div>
                        <button type="submit">处理文件</button>
                    </form>

                    <div id="loading" class="loading">
                        <div class="spinner"></div>
                        处理文件中...这可能需要一些时间，取决于文件大小和内容。
                    </div>

                    <div id="results" class="results">
                        <h2>处理结果</h2>
                        <div id="resultsSummary"></div>
                        <pre id="resultsJson"></pre>
                        <button id="downloadBtn" class="download-btn">下载结果 (JSON)</button>
                    </div>

                    <script>
                        // 显示选择的文件
                        document.getElementById('files').addEventListener('change', function(e) {
                            const fileList = document.getElementById('fileList');
                            fileList.innerHTML = '';

                            if (this.files.length > 0) {
                                for (let i = 0; i < this.files.length; i++) {
                                    const file = this.files[i];
                                    const fileItem = document.createElement('div');
                                    fileItem.className = 'file-item';
                                    fileItem.textContent = file.name + ' (' + (file.size / 1024).toFixed(2) + ' KB)';
                                    fileList.appendChild(fileItem);
                                }
                            }
                        });

                        // 表单提交处理
                        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                            e.preventDefault();

                            const formData = new FormData(this);
                            const overlap = document.getElementById('overlap').checked;
                            formData.set('overlap', overlap.toString());

                            document.getElementById('loading').style.display = 'block';
                            document.getElementById('results').style.display = 'none';

                            try {
                                const response = await fetch('/upload', {
                                    method: 'POST',
                                    body: formData
                                });

                                if (!response.ok) {
                                    throw new Error('服务器响应错误: ' + response.status);
                                }

                                const results = await response.json();

                                // 显示结果摘要
                                const summaryDiv = document.getElementById('resultsSummary');
                                summaryDiv.innerHTML = '';

                                let totalSegments = 0;
                                let fileCount = 0;

                                for (const filename in results) {
                                    fileCount++;
                                    const segments = results[filename];
                                    const segmentCount = Array.isArray(segments) ? segments.length : 0;
                                    totalSegments += segmentCount;

                                    const fileResult = document.createElement('p');
                                    fileResult.innerHTML = `<strong>${filename}</strong>: ${segmentCount} 个段落`;
                                    summaryDiv.appendChild(fileResult);
                                }

                                const totalSummary = document.createElement('p');
                                totalSummary.innerHTML = `<strong>总计:</strong> ${fileCount} 个文件, ${totalSegments} 个段落`;
                                summaryDiv.appendChild(totalSummary);

                                // 显示详细JSON结果
                                document.getElementById('resultsJson').textContent = JSON.stringify(results, null, 2);
                                document.getElementById('results').style.display = 'block';
                                document.getElementById('downloadBtn').style.display = 'inline-block';

                                // 存储当前结果用于下载
                                window.currentResults = results;

                            } catch (error) {
                                console.error('错误:', error);
                                document.getElementById('resultsSummary').innerHTML = `<p style="color: red;">处理文件失败: ${error.message}</p>`;
                                document.getElementById('resultsJson').textContent = JSON.stringify({ error: '处理文件失败' }, null, 2);
                                document.getElementById('results').style.display = 'block';
                                document.getElementById('downloadBtn').style.display = 'none';
                            } finally {
                                document.getElementById('loading').style.display = 'none';
                            }
                        });

                        // 下载结果
                        document.getElementById('downloadBtn').addEventListener('click', async function() {
                            if (!window.currentResults) return;

                            try {
                                const response = await fetch('/api/download_results', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json',
                                    },
                                    body: JSON.stringify(window.currentResults)
                                });

                                if (!response.ok) {
                                    throw new Error('下载失败: ' + response.status);
                                }

                                // 创建一个blob链接并点击下载
                                const blob = await response.blob();
                                const url = window.URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.style.display = 'none';
                                a.href = url;
                                a.download = 'document_parser_results.json';
                                document.body.appendChild(a);
                                a.click();
                                window.URL.revokeObjectURL(url);

                            } catch (error) {
                                console.error('下载错误:', error);
                                alert('下载结果失败: ' + error.message);
                            }
                        });
                    </script>
                </body>
                </html>
                ''')

            # 启动Flask服务器
            app.run(host=args.host, port=args.port, debug=args.debug)

    except Exception as e:
        print(f"处理文档时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()