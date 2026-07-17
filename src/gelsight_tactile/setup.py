from setuptools import find_packages, setup

package_name = 'gelsight_tactile'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='potato',
    maintainer_email='jys020829@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "camera_viewer = gelsight_tactile.camera_viewer:main",
            "collect_calib_data=gelsight_tactile.collect_calib_data:main",
            "poisson_solver=gelsight_tactile.poisson_solver:main",
            "reconstructor_node=gelsight_tactile.reconstructor_node:main",
            "reconstructor=gelsight_tactile.reconstructor:main",
            "visualizer3d=gelsight_tactile.visualizer3d:main",

        ],
    },
)
