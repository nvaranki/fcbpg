package com.varankin.bpg;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * @author &copy; 2024 Nikolai Varankine
 */
final class Model
{
    final float[][] w, q;
    final int sb;

    Model( int sx, int sh, int sy, boolean bias )
    {
        sb = bias ? 1 : 0;
        w = new float[sx+sb][sh];
        q = new float[sh+sb][sy];
    }

    void reset( RandomGenerator rnd )
    {
        for( float[] a : w )
            for( int i = 0; i < a.length; i++ )
                a[i] = ( rnd.nextFloat() * 2F - 1F ) * 0.001F;
        for( float[] a : q )
            for( int i = 0; i < a.length; i++ )
                a[i] = ( rnd.nextFloat() * 2F - 1F ) * 0.001F;
    }

    void train( float[] x, float[] h, float[] e, int lri )
    {
        float[][] dq = backward0( h, q, e );
        float[][] dw = backward1( x, w, q, e );
        int n = 2* lri;
        sum( q, dq, n );
        sum( w, dw, n );
    }

    float[] infer( float[] x, float[] y )
    {
        float[] h = new float[q.length];
        Arrays.fill( h, q.length-sb, q.length, 1F );
        forward( x, w, h );
        //relu( h );
        forward( h, q, y );
        return h;
    }

    private float[][] backward1( float[] x, float[][] w, float[][] q, float[] e )
    {
        float[][] dw = new float[w.length][q.length-sb];
        for( int ix = 0; ix < x.length; ix++ )
            for( int iq = 0; iq < q.length-sb; iq++ )
            {
                float s = 0F;
                for( int ie = 0; ie < e.length; ie++ )
                    s += e[ie] / q[iq][ie];
                dw[ix][iq] = s / x[ix] / x.length; //TODO Float.NaN Float.*_INFINITY
                if( ! Float.isFinite( dw[ix][iq] ) )
                    dw[ix][iq] = 0F;
            }
        return dw;
    }

    private float[][] backward0( float[] x, float[][] w, float[] e )
    {
        float[][] dw = new float[w.length][e.length];
        for( int ix = 0; ix < x.length; ix++ )
            for( int ie = 0; ie < e.length; ie++ )
            {
                dw[ix][ie] = e[ie] / x[ix] / x.length; //TODO Float.NaN Float.*_INFINITY
                if( ! Float.isFinite( dw[ix][ie] ) )
                    dw[ix][ie] = 0F;
            }
        return dw;
    }

    private void forward( float[] a, float[][] w, float[] r )
    {
        for( int ir = 0; ir < r.length - sb; ir++ )
        {
            float v = 0F;
            for( int ia = 0; ia < a.length; ia++ )
                v += a[ia] * w[ia][ir];
            r[ir] = v;
        }
    }

    private static void sum( float[][] a, float[][] da, float n )
    {
        for( int ir = 0; ir < a.length; ir++ )
            for( int ic = 0; ic < a[ir].length; ic++ )
                a[ir][ic] += da[ir][ic] / n;
    }

    static void relu( float[] m )
    {
        for( int i = 0; i < m.length; i++ )
            m[i] = Math.max( 0F, m[i] );
    }

}
