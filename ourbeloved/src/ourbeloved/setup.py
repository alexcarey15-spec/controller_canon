from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'ourbeloved'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    (os.path.join('share', package_name, 'config'), glob('config/*')),
],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts':
        [
         "cannon_node = ourbeloved.cannon_node:main",
         "cartesian_ptp_node = ourbeloved.cartesian_ptp_node:main",
         "wx250s_kinematics = ourbeloved.wx250s_kinematics:main",
         "joint_state_node = ourbeloved.joint_state_node:main",
         "joy_input_node = ourbeloved.joy_input_node:main",
         "pose_node = ourbeloved.pose_node:main",
        ],
    },
)
