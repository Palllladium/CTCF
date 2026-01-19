from pathlib import Path

def main():
    # ✅ Сканируем папку, где лежит сам скрипт
    script_dir = Path(__file__).resolve().parent
    current_dir = script_dir

    output_file = current_dir / "PROJECT_STRUCTURE.txt"

    print(f"📁 Сканируем: {current_dir}")
    print(f"💾 Сохраняем в: {output_file}")

    # Игнорируемые элементы
    ignored = {
        '.git', '__pycache__', '.pytest_cache', '.vscode', '.idea',
        'node_modules', 'venv',
        '.DS_Store', 'Thumbs.db',
        'parse.py',  # сам парсер
        'PROJECT_STRUCTURE.txt',
        'plus_api_by_deep.md',
        'architecture.pdf'
    }

    structure = []
    contents = []

    def read_file_smart(filepath: Path) -> str:
        filename = filepath.name.lower()

        # Особые файлы - сначала пробуем UTF-16 LE
        if filename in ['requirements.txt', 'readme.md']:
            for encoding in ['utf-16-le', 'utf-16', 'utf-8', 'cp1251']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                        if content.startswith('\ufeff'):
                            content = content[1:]
                        print(f"  ✓ {filepath.relative_to(current_dir)} ({encoding})")
                        return content
                except Exception:
                    continue
            return f"[Не удалось прочитать {filename} ни в одной кодировке]"

        text_exts = {'.py', '.txt', '.md', '.json', '.yml', '.yaml',
                     '.html', '.css', '.js', '.xml', '.ini', '.cfg'}

        if filepath.suffix.lower() in text_exts:
            for encoding in ['utf-8', 'cp1251']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        return f.read()
                except Exception:
                    continue
            return "[Бинарный файл или ошибка кодировки]"

        return "[Бинарный файл]"

    def build_tree(dir_path: Path, prefix: str = ""):
        try:
            items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except Exception:
            return

        # фильтр по ignored
        dirs = [i for i in items if i.is_dir() and i.name not in ignored]
        files = [i for i in items if i.is_file() and i.name not in ignored]

        combined = dirs + files

        for i, item in enumerate(combined):
            is_last = (i == len(combined) - 1)

            if item.is_dir():
                structure.append(f"{prefix}{'└── ' if is_last else '├── '}📁 {item.name}/")
                build_tree(item, prefix + ("    " if is_last else "│   "))
            else:
                structure.append(f"{prefix}{'└── ' if is_last else '├── '}📄 {item.name}")

                rel_path = item.relative_to(current_dir)
                content = read_file_smart(item)
                if len(content) > 50000:
                    content = content[:50000] + "\n[... обрезано ...]"

                contents.append({
                    "path": rel_path.as_posix(),
                    "content": content
                })

    structure.append(f"📁 {current_dir.name}/")
    build_tree(current_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ПОЛНАЯ СТРУКТУРА ПРОЕКТА\n")
        f.write(f"Директория: {current_dir}\n")
        f.write("=" * 100 + "\n\n")

        f.write("ДРЕВОВИДНАЯ СТРУКТУРА:\n")
        f.write("-" * 50 + "\n")
        f.write("\n".join(structure))
        f.write("\n\n" + "=" * 100 + "\n\n")

        f.write("СОДЕРЖИМОЕ ФАЙЛОВ:\n")
        f.write("=" * 100 + "\n\n")

        for item in contents:
            f.write(f"\n📄 {item['path']}\n")
            f.write("-" * 50 + "\n")
            f.write(item["content"])
            f.write("\n" + "=" * 50 + "\n")

    print(f"\n✅ Готово! Файл сохранен: {output_file}")
    print(f"📊 Строк структуры: {len(structure)}")
    print(f"📄 Файлов обработано: {len(contents)}")

if __name__ == "__main__":
    main()