from setuptools import find_packages, setup

setup(
    name="deep_fake_voice_detection",
    packages=[
        package
        for package in find_packages()
        if package.startswith("deep_fake_voice_detection")
    ],
)
