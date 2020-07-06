"""
Test indexing and updating of tile coding
"""
from numpy.testing import assert_almost_equal

from rlmarket.agent.q_function import TileCodedFunction


def test_tile_creation():
    """ Test tile creation """
    tiles = TileCodedFunction(state_dimension=2, num_actions=3, num_tiles=5, granularity=10)

    # Test tile creation
    assert len(tiles.tables) == 5
    assert len(tiles.bin_groups) == 5
    assert_almost_equal(tiles.bin_groups[0], tiles.bin_groups[2] - 0.24, decimal=10)
    assert_almost_equal(tiles.bin_groups[1], tiles.bin_groups[2] - 0.12, decimal=10)
    assert_almost_equal(tiles.bin_groups[2], (-3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3), decimal=10)
    assert_almost_equal(tiles.bin_groups[3], tiles.bin_groups[2] + 0.12, decimal=10)
    assert_almost_equal(tiles.bin_groups[4], tiles.bin_groups[2] + 0.24, decimal=10)
    # bin 0 and bin 4 should be only "0.12 apart"
    assert_almost_equal((tiles.bin_groups[4] + 0.12)[:-1], tiles.bin_groups[0][1:], decimal=10)


def test_tile_indexing():
    """ Test tile indexing """
    tiles = TileCodedFunction(state_dimension=2, num_actions=3, num_tiles=2, granularity=6)
    assert tuple(tiles[-3.25, 2.749]) == (0, 0, 0)
    assert tuple(tiles[-3.251, 2.75]) == (0, 0, 0)
    assert tuple(tiles[-2.75, 2.25]) == (0, 0, 0)
    assert tuple(tiles[-2.25, 0]) == (0, 0, 0)
    assert tuple(tiles.tables[0].table.keys()) == ((1, 6), (0, 7), (2, 4))
    assert tuple(tiles.tables[1].table.keys()) == ((0, 6), (1, 6), (1, 3))


def test_tile_update():
    """ Test tile value udpate """
    tiles = TileCodedFunction(state_dimension=2, num_actions=3, num_tiles=2, granularity=6)
    tiles.update((0, 0), 1, 1, 0.1)
    assert tuple(tiles.tables[0].table.keys()) == ((4, 4),)
    assert tuple(tiles.tables[0].table[4, 4]) == (0, 0.1, 0)
    assert tuple(tiles.tables[1].table.keys()) == ((3, 3),)
    assert tuple(tiles.tables[1].table[3, 3]) == (0, 0.1, 0)
