package com.varankin.bpg;

import com.varankin.bpg.DataSource.IO;

import java.util.AbstractCollection;
import java.util.Arrays;
import java.util.Iterator;

/**
 * @author &copy; 2024 Nikolai Varankine
 */
final class DataArray extends AbstractCollection<IO> implements DataSource
{
    private final IO[] io;

    DataArray( int n, boolean bias )
    {
        io = new IO[n];
        for( int i = 0; i < io.length; i++ )
        {
            float[] inp = new float[]
            {
                    ( i & 0b0001 ) > 0 ? 1F : 0F,
                    ( i & 0b0010 ) > 0 ? 1F : 0F,
                    ( i & 0b0100 ) > 0 ? 1F : 0F,
                    ( i & 0b1000 ) > 0 ? 1F : 0F,
                    //TODO n
            };

            float v = i < io.length / 2 ? 1F : 0F;
            float[] out = new float[]{ v, 1F - v };

            if( bias )
            {
                float[] ext = Arrays.copyOf( inp, inp.length + ( bias ? 1 : 0 ) );
                Arrays.fill( ext, inp.length, ext.length, 1F );
                inp = ext;
            }
            io[i] = new IO( inp, out );
        }
    }

    @Override
    public Iterator<IO> iterator()
    {
        return Arrays.asList( io ).iterator();
    }

    @Override
    public int size()
    {
        return io.length;
    }
}
