# Assignment 3 - DCGAN on MNIST

This implementation follows the assignment brief in `DLP_Assignment_3.pdf`:
- Uses a **DCGAN** architecture with separate Generator and Discriminator.
- Trains on the **MNIST** handwritten digit dataset.
- Produces generated sample images and at least one output with **10 digits**.

## Environment

Recommended for your hardware (RTX 3060 Ti):
- Python 3.10+
- PyTorch with CUDA support
- 8GB VRAM-friendly defaults are already configured

Install dependencies:

```bash
pip install torch torchvision matplotlib
```

## Run Training

From the `A03` directory:

```bash
python dcgan_mnist.py --epochs 30 --batch-size 128 --out-dir outputs
```

Useful fast test run:

```bash
python dcgan_mnist.py --epochs 2 --batch-size 128 --out-dir outputs_quick
```

## Outputs

After training, files are saved under your output folder:
- `samples/epoch_XXX.png` - periodic sample grids during training
- `checkpoints/dcgan_epoch_XXX.pt` - model checkpoints
- `loss_curve.png` - training losses
- `generated_10_digits.png` - final strip with 10 generated digits

## Notes

- Mixed precision (`AMP`) is enabled by default for faster GPU training.
- Disable AMP if needed:

```bash
python dcgan_mnist.py --no-amp
```
