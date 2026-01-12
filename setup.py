from setuptools import find_packages, setup

package_name = 'pr_vr_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Install package manifest and include the vr_pose_client.py so it is
        # available in the installed share folder (e.g. share/pr_vr_teleop).
        ('share/' + package_name, ['package.xml', 'pr_vr_teleop/vr_pose_client.py']),
        ('lib/' + package_name, ['pr_vr_teleop/vr_pose_client.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='josyula',
    maintainer_email='krishna.j@perceptyne.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "pr_vr_node = pr_vr_teleop.pr_vr_node:main",
        ],
    },
)
