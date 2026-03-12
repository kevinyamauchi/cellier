from cellier.utils.chunked_image import MultiscaleImageModel


def test_multiscale_image_model():
    """Test instantiating a MultiscaleImageModel."""
    model = MultiscaleImageModel.from_shape_and_scales(
        shape=(100, 100, 100),
        chunk_shapes=[(25, 25, 25), (25, 25, 25)],
        downscale_factors=[1.0, 2.0],
    )

    # check there are the correct number of scales
    assert model.n_scales == 2

    # check scale 0
    assert model.scales[0].chunk_grid_shape == (4, 4, 4)

    # check scale 1
    assert model.scales[1].chunk_grid_shape == (2, 2, 2)
