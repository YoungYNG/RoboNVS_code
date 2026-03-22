from pathlib import Path
import re
from setuptools import setup, find_packages

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = ROOT / "requirements.txt"
DA3_PYPROJECT_PATH = ROOT / "Depth-Anything-3" / "pyproject.toml"


def normalize_name(requirement: str) -> str:
    requirement = requirement.strip()
    if not requirement or requirement.startswith("#"):
        return ""
    if " @ " in requirement:
        name = requirement.split(" @ ", 1)[0]
    else:
        match = re.match(r"^[A-Za-z0-9_.-]+", requirement)
        name = match.group(0) if match else requirement
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirements_txt(path: Path) -> list[str]:
    if not path.exists():
        return []
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


def read_da3_dependencies(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    if not path.exists():
        return [], {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    dependencies = list(project.get("dependencies", []))
    optional_dependencies = {
        key: list(value) for key, value in project.get("optional-dependencies", {}).items()
    }
    return dependencies, optional_dependencies


def merge_requirements(*groups: list[str], overrides: dict[str, str] | None = None) -> list[str]:
    merged: dict[str, str] = {}
    order: list[str] = []

    for group in groups:
        for requirement in group:
            name = normalize_name(requirement)
            if not name:
                continue
            if name not in merged:
                order.append(name)
            merged[name] = requirement

    if overrides:
        for name, requirement in overrides.items():
            normalized = normalize_name(name)
            if normalized not in merged:
                order.append(normalized)
            merged[normalized] = requirement

    return [merged[name] for name in order]


root_requirements = read_requirements_txt(REQUIREMENTS_PATH)
da3_requirements, da3_optional = read_da3_dependencies(DA3_PYPROJECT_PATH)

compatibility_overrides = {
    "huggingface-hub": "huggingface-hub>=0.30.0,<1.0",
    "moviepy": "moviepy==1.0.3",
    "typer": "typer>=0.9.0",
    "numpy": "numpy<2",
    "opencv-python": "opencv-python<4.13",
    "opencv-python-headless": "opencv-python-headless<4.13",
    "pillow-heif": "pillow-heif",
}

base_requirements = merge_requirements(
    root_requirements,
    da3_requirements,
    da3_optional.get("app", []),
    overrides=compatibility_overrides,
)

extras_require = {
    "da3-gs": merge_requirements(da3_optional.get("gs", [])),
    "da3-all": merge_requirements(da3_optional.get("app", []), da3_optional.get("gs", [])),
}

setup(
    name="robonvs",
    version="1.1.1",
    description="Enjoy the magic of Diffusion models!",
    author="Artiprocher",
    packages=find_packages(exclude=["diffsynth", "diffsynth.*"]),
    install_requires=base_requirements,
    extras_require=extras_require,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_data={"diffsynth": ["tokenizer_configs/**/**/*.*"]},
    python_requires=">=3.9,<=3.13",
)
