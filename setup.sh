#!/bin/bash

echo "Setting up Crypto Signal Analysis System..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# OpenAI API Configuration
OPENAI_API_KEY=your-api-key-here

# Monitoring Settings
CHECK_INTERVAL=300
OUTPUT_DIR=signal_reports
EOL
    echo "✅ .env file created. Please add your OpenAI API key."
else
    echo "ℹ️  .env file already exists."
fi

# Create output directory
mkdir -p signal_reports

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use the system:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the system: python main.py"
echo ""
echo "Options:"
echo "  python main.py --test           # Run single test"
echo "  python main.py --interval 60    # Check every 60 seconds"
echo "  python main.py --help           # Show all options"