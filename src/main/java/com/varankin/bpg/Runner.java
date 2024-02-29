package com.varankin.bpg;

import java.util.Random;

public class Runner
{

    public static void main( String... args )
    {
        Runner runner = new Runner();
        float loss;
        do
        {
            loss = runner.epoch();
            System.out.printf( "Epoch loss: %5.3f%n", loss ); //TODO
        }
        while( loss > 0.001F ); //TODO
        System.out.println( "Finished." );
    }

    private record IO( float[] inp, float[] out ) {}

    private final float[][] w, q;
    private final IO[] io;

    public Runner()
    {
        w = new float[4][3];
        q = new float[3][2];
        Random rnd = new Random( 12345L );
        for( float[] a : w )
            for( int i = 0; i < a.length; i++ )
                a[i] = rnd.nextFloat() * 2F - 1F;
        for( float[] a : q )
            for( int i = 0; i < a.length; i++ )
                a[i] = rnd.nextFloat() * 2F - 1F;

        io = new IO[16];
        for( int i = 0; i < io.length; i++ )
        {
            float[] inp = new float[]
            {
                ( i & 0b0001 ) > 0 ? 1F : 0F,
                ( i & 0b0010 ) > 0 ? 1F : 0F,
                ( i & 0b0100 ) > 0 ? 1F : 0F,
                ( i & 0b1000 ) > 0 ? 1F : 0F,
            };

            float v = i < io.length / 2 ? 1F : 0F;
            float[] out = new float[]{ v, 1F - v };

            io[i] = new IO( inp, out );
        }
    }

    private float epoch()
    {
        float[] y = new float[2];
        double loss = 0F;
        for( IO p : io )
        {
            float[] x = p.inp();
            float[] h = infer( x, y );

            float[] t = p.out();
            float[] e = new float[y.length];
            for( int i = 0; i < e.length; i++ )
            {
                e[i] = t[i] - y[i];
                loss += e[i] * e[i]; //TODO max ?
            }
            train( x, h, e );
        }
        return (float) Math.sqrt( loss / io.length );
    }

    private static float[][] backward1( float[] x, float[][] w, float[][] q, float[] e )
    {
        float[][] dw = new float[x.length][q.length];
        for( int ix = 0; ix < x.length; ix++ )
            for( int iq = 0; iq < q.length; iq++ )
            {
                float s = 0F;
                for( int ie = 0; ie < e.length; ie++ )
                    s += e[ie] / q[iq][ie];
                dw[ix][iq] = s / x[ix] / e.length; //TODO Float.NaN Float.*_INFINITY
                if( ! Float.isFinite( dw[ix][iq] ) )
                    dw[ix][iq] = 0F;
            }
        return dw;
    }

    private static float[][] backward0( float[] x, float[][] w, float[] e )
    {
        float[][] dw = new float[x.length][e.length];
        for( int ix = 0; ix < x.length; ix++ )
            for( int ie = 0; ie < e.length; ie++ )
            {
                dw[ix][ie] = e[ie] / x[ix] / e.length; //TODO Float.NaN Float.*_INFINITY
                if( ! Float.isFinite( dw[ix][ie] ) )
                    dw[ix][ie] = 0F;
            }
        return dw;
    }

    private static void forward( float[] a, float[][] w, float[] r )
    {
        for( int ir = 0; ir < r.length; ir++ )
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

    private void train( float[] x, float[] h, float[] e )
    {
        float[][] dq = backward0( h, q, e );
        float[][] dw = backward1( x, w, q, e );
        sum( q, dq, 2 );
        sum( w, dw, 2 );
    }

    private float[] infer( float[] x, float[] y )
    {
        float[] h = new float[q.length];
        forward( x, w, h );
        forward( h, q, y );
        return h;
    }

}
