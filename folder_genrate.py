import os

def create_sentiment_ai_structure():
    """
    Creates the complete folder structure for the sentiment_ai project
    """
    
    # Define the project structure
    structure = {
        'sentiment_ai': {
            'data': {
                'raw': ['feedback.csv'],
                'processed': ['train.csv', 'val.csv', 'test.csv']
            },
            'configs': ['local.yaml', 'cloud.yaml', 'labels.yaml'],
            'src': {
                'data': ['loader.py', 'cleaner.py', 'splitter.py'],
                'features': ['tokenizer.py'],
                'models': ['model_loader.py'],
                'training': ['train.py', 'loss.py', 'trainer.py'],
                'evaluation': ['metrics.py', 'confusion.py'],
                'inference': ['predictor.py'],
                'utils': ['logger.py']
            },
            'artifacts': {
                'models': [],
                'metrics': [],
                'logs': []
            },
            '__files__': ['requirements.txt', 'README.md']
        }
    }
    
    def create_structure(base_path, structure_dict):
        """
        Recursively creates directories and files
        """
        for key, value in structure_dict.items():
            if key == '__files__':
                # Create files in the current directory
                for filename in value:
                    filepath = os.path.join(base_path, filename)
                    with open(filepath, 'w') as f:
                        if filename == 'README.md':
                            f.write('# Sentiment AI Project\n\nSentiment analysis project structure.')
                        elif filename == 'requirements.txt':
                            f.write('# Add your project dependencies here\n')
                    print(f"Created file: {filepath}")
            else:
                # Create directory
                dir_path = os.path.join(base_path, key)
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
                
                if isinstance(value, dict):
                    # Recursively create subdirectories
                    create_structure(dir_path, value)
                elif isinstance(value, list):
                    # Create files in this directory
                    for filename in value:
                        filepath = os.path.join(dir_path, filename)
                        with open(filepath, 'w') as f:
                            if filename.endswith('.py'):
                                f.write(f'# {filename}\n')
                            elif filename.endswith('.yaml'):
                                f.write(f'# Configuration file\n')
                            elif filename.endswith('.csv'):
                                f.write('# CSV data file\n')
                        print(f"Created file: {filepath}")
    
    # Create the structure
    print("Creating sentiment_ai project structure...\n")
    create_structure('.', structure)
    print("\n✓ Project structure created successfully!")

if __name__ == "__main__":
    create_sentiment_ai_structure()