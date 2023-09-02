# Knotoids

Knotoids is a command-line tool that computes the knotoid distribution of a 3D piecewise linear (PL) curve.

## Background

Knotoids are a mathematical abstraction extending traditional knot theory to include curves with open ends, offering a more natural model for studying 3D structures like proteins. Unlike knots, the class of a knotoid is sensitive to the angle of its 2D projection from 3D space. Traditionally, Monte Carlo (MC) methods have been used for approximating a distribution of knotoids for such curves, as implemented in the Knoto-ID program.

This project, Knotoids, takes a deterministic approach. It precisely computes the knotoid distribution for a given piecewise linear curve in 3D space by partitioning the sphere of possible 2D projections into regions that share a knotoid class. This removes the need for approximations, providing an exact classification. It also dramatically reduces the number knotoid classifications to the number of regions on the sphere, rather than the number of MC samples, thus improving performance.

## Requirements

- Python 3.x
- Poetry
- [Knoto-ID](https://github.com/sib-swiss/Knoto-ID)

## Installation

### Using Poetry

1. If you haven't installed Poetry yet, follow the installation steps provided on [Poetry's official website](https://python-poetry.org/docs/#installation).

2. Clone the Knotoids repository to your local system:

```bash
git clone https://github.com/nyerolemou/knotoids.git
cd knotoids
```

3. Once inside the project directory, run the following command to install the required dependencies and set up a virtual environment:

```bash
poetry install
```

### Knoto-ID Setup

Knoto-ID is a crucial dependency for Knotoids. You can either compile it from source or download a pre-built executable from its [GitHub repository](https://github.com/sib-swiss/Knoto-ID). Specify its location using the `-k` option when running Knotoids.

## Usage

### Basic Example

Here's a quick example to demonstrate the basic usage:

```bash
knotoids -s path/to/curve.xyz -o send/output/here -k path/to/knoto-id
```

### Command Line Options

For a complete list of available command-line options, execute:

```bash
knotoids --help
```

Output:

```bash
Usage: knotoids [OPTIONS]                                                                                                                              
                                                                                                                                                        
 Compute the knotoid distribution of a piecewise linear curve.                                                                                          
                                                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --source              -s      FILE                             Path to the PL curve coordinates. [default: None] [required]                       │
│ *  --knoto-id            -k      DIRECTORY                        Path to Knoto-ID root directory. [default: None] [required]                        │
│    --output              -o      DIRECTORY                        Path to output directory. [default: None]                                          │
│    --verbose             -v                                       If True, save all region data.                                                     │
│    --install-completion          [bash|zsh|fish|powershell|pwsh]  Install completion for the specified shell. [default: None]                        │
│    --show-completion             [bash|zsh|fish|powershell|pwsh]  Show completion for the specified shell, to copy it or customize the installation. │
│                                                                   [default: None]                                                                    │
│    --help                                                         Show this message and exit.                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Input Formats

The tool accepts input coordinates in the form of `.npy`, `.txt`, or `.xyz` files.

### Verbose Mode

For a detailed output, which includes all regional data generated during the computation, use the `-v` or `--verbose` flag:

```bash
knotoids -s path/to/curve.xyz -o send/output/here -k path/to/knoto-id -v
```

### Output

If you don't specify an `output` directory, Knotoids will print a summary of the knotoid distribution to stdout and launch an interactive Plotly plot via your default web browser.

If you do provide an `output` directory, Knotoids will save the following files:

- `output/summary.json`: A JSON file summarizing the knotoid distribution.
- `output/knotoid_distribution.html`: An interactive Plotly HTML plot representing the knotoid distribution.

## Licensing

The code in this repository is licensed under the MIT license. For more details, please refer to the `LICENSE` file in the repository.

## Contributors

Developed by Naya Yerolemou.

The idea for this project arose from a collaboration with Agnese Barbensi and Oliver Vipond.

Thanks to Alex Thorne for constructive feedback on the code.

Special thanks go to [Knoto-ID](https://github.com/sib-swiss/Knoto-ID) by Julien Dorier & Dimos Goundaroulis for handling the knotoid classification.
