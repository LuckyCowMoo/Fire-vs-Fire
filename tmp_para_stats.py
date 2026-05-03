import os, docx, sys

folder = r'C:\Users\lukea\Desktop\Stuff\School\Year 3\Project\Fire v Fire\AI texts'
MIN = 10
all_lengths = []

for fname in ['Claude.docx', 'Gemeni.docx', 'GPT.docx', 'Grock.docx', 'llama.docx']:
    doc = docx.Document(os.path.join(folder, fname))
    lengths = [len(p.text.strip()) for p in doc.paragraphs
               if len(p.text.strip()) >= MIN and p.text.strip() != '</p><p>']
    avg = int(sum(lengths) / len(lengths)) if lengths else 0
    sys.stdout.write(f'{fname}: {len(lengths)} paragraphs >= {MIN} chars, avg = {avg} chars\n')
    all_lengths.extend(lengths)

overall_avg = int(sum(all_lengths) / len(all_lengths)) if all_lengths else 0
sys.stdout.write(f'\nOverall: {len(all_lengths)} paragraphs, avg = {overall_avg} chars\n')
sys.stdout.flush()
