import os
from glob import glob
from setuptools import setup

package_name = 'yolov3_ros'
share_dir = 'share/' + package_name

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (share_dir, ['package.xml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jinseog',
    maintainer_email='jstar0525@gmail.com',
    description='YOLOv3 pytorch ros',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = yolov3_ros.detector:main'
        ],
    },
)
