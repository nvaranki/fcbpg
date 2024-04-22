package com.varankin.bpg;

import java.util.Collection;

/**
 * @author &copy; 2024 Nikolai Varankine
 */
interface DataSource extends Collection<DataSource.IO>
{
    record IO( float[] inp, float[] out )
    {
    }
}
