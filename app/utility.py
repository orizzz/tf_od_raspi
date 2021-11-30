import re, json, cv2
from config import Config

def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def draw_lines(frame):
  LINE = json.loads(Config.LINE)
  for item in LINE:
    start = (item["line"][0][0], item["line"][0][1])
    end = (item["line"][1][0], item["line"][1][1])
    cv2.line(frame, start, end, (0,255,0), 3)