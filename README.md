# COVID Monitor

The COVID Monitor is a computer vision application designed to detect mask usage and monitor social distancing using pre-trained deep learning models. This application uses SSD MobileNet for person detection and EfficientDet for mask detection in video feeds, making it ideal for enhancing public health compliance in crowded settings.

## Project Features
- **Mask Detection**: Identifies whether individuals in a video are wearing face masks.
- **Social Distancing Monitoring**: Measures the distance between individuals to ensure social distancing guidelines are met.
- **Real-Time Processing**: Analyzes video feeds in real-time to provide immediate compliance information.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Python 3.8+
- Pip

### Installation
Clone this repository to your local machine:

```bash
git clone https://github.com/AngelikaPolshikova/Covide_monitor.git
cd Covide_monitor

Install the required dependencies:

pip install -r requirements.txt


### Usage
Run the main application script from the command line:

bash
Copy code
python src/detection_script.py