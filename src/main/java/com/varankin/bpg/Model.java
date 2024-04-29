package com.varankin.bpg;

import gov.nist.math.Jama.Matrix;
import gov.nist.math.Jama.SingularValueDecomposition;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * @author &copy; 2024 Nikolai Varankine
 */
final class Model
{
    final float[][] w, q, dw, dq;
    final float[] x, h, y;
    private final int sb;

    Model( int sx, int sh, int sy, boolean bias )
    {
        sb = bias ? 1 : 0;
        x  = new float[sx+sb];
        w  = new float[sx+sb][sh];
        dw = new float[sx+sb][sh];
        h  = new float[sh+sb];
        q  = new float[sh+sb][sy];
        dq = new float[sh+sb][sy];
        y  = new float[sy+sb];
        // preset constant bias
        Arrays.fill( x, sx, sx+sb, 1F );
        Arrays.fill( h, sh, sh+sb, 1F );
        Arrays.fill( y, sy, sy+sb, 1F );
    }

    void reset( RandomGenerator rnd )
    {
        for( float[] a : w )
            for( int i = 0; i < a.length; i++ )
                a[i] = ( rnd.nextFloat() * 2F - 1F ) / ( w.length * a.length );
        for( float[] a : q )
            for( int i = 0; i < a.length; i++ )
                a[i] = ( rnd.nextFloat() * 2F - 1F ) / ( q.length * a.length );
    }

    void train( float[] e, int lrd )
    {
        backward( h, dq, (r,c) -> e0( r, c, e, h ) );
        backward( x, dw, (r,c) -> e1( r, c, e, x ) );
        float lr = 1f / ( 2 * lrd ); // 2 sums
        add( q, dq, lr );
        add( w, dw, lr );
        return;
    }

    private static float rated( float e, float[] a, float[][] m, int r, int c )
    {
        float f = Float.NaN, d = 0f;
        for( int i = 0; i < m.length; i++ )
        {
            float v = Math.abs( m[i][c] * a[i] );
            d += v;
            if( i == r )
                f = v;
        }
        return e * f / d;
    }

    private float e0( int r, int c, float[] e, float[] h )
    {
        return rated( e[c], h, q, r, c );
    }

    private float e1( int r, int c, float[] e, float[] h )
    {
        float s = 0F;
        for( int i = 0; i < e.length; i++ )
            s += e[i] / q[c][i];
        return rated( s, x, w, r, c );
    }

    void infer( float[] input )
    {
        System.arraycopy( input, 0, x, 0, x.length );
        forward( x, w, h );
        //relu( h );
        forward( h, q, y );
    }

    @FunctionalInterface
    private interface OutError
    {
        float of( int r, int c );
    }

    private static void backward( float[] v, float[][] d, OutError e )
    {
        for( int ir = 0; ir < d.length; ir++ )
            for( int ic = 0; ic < d[ir].length; ic++ )
            {
                d[ir][ic] = e.of( ir, ic ) / v[ir]; //TODO Float.NaN Float.*_INFINITY
                if( ! Float.isFinite( d[ir][ic] ) )
                    d[ir][ic] = 0F;
            }
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

    private static void add( float[][] a, float[][] da, float lr )
    {
        for( int ir = 0; ir < a.length; ir++ )
            for( int ic = 0; ic < a[ir].length; ic++ )
                a[ir][ic] += da[ir][ic] * lr;
    }

    static void relu( float[] m )
    {
        for( int i = 0; i < m.length; i++ )
            m[i] = Math.max( 0F, m[i] );
    }

    @Deprecated
    double[][][] decompose()
    {
        double[][] wd = new double[w.length][];
        for( int r = 0; r < w.length; r++ )
        {
            float[] a = w[r];
            wd[r] = new double[a.length];
            for( int c = 0; c < a.length; c++ )
                wd[r][c] = a[c];
        }
        SingularValueDecomposition svd = new SingularValueDecomposition( new Matrix( wd ) );
        return new double[][][]
                {
                        svd.getU().getArray(),
                        svd.getS().getArray(),
                        svd.getV().getArray(),
                };
    }

}
