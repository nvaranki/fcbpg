package com.varankin.bpg;

import com.varankin.bpg.DataSource.IO;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Random;

/**
 * @author &copy; 2024 Nikolai Varankine
 */
public final class Runner
{

    public static void main( String... args )
    {
        Runner runner = new Runner( 4, 8, 2, true );
        runner.model.reset( new Random( 12345L ) );
        runner.run( 0.050F );
    }

    private final Model model;
    private final DataSource io;

    private Runner( int sx, int sh, int sy, boolean bias )
    {
        model = new Model( sx, sh, sy, bias );
        io = new DataArray( 16, bias );
    }

    private void run( float accuracy )
    {
        float loss;
        int n = 0;
        do
        {
            // 4*16*2 schema:
            // *100   Epoch #550331 loss: 0,049
            // *1000  Epoch #2081 loss: 0,041
            // *10000 Epoch #15927 loss: 0,050
            loss = epoch( 2, 100 );
            System.out.printf( "Epoch #%d loss: %5.3f%n", n++, loss ); //TODO
        }
        while( Float.isFinite( loss ) && loss > accuracy ); //TODO
        System.out.println( "Finished." );
        try( OutputStream s = new FileOutputStream( "data/model_w.txt" ) )
        {
            save( model.w, new OutputStreamWriter( s, StandardCharsets.UTF_8 ) );
        }
        catch( IOException ex )
        {
            ex.printStackTrace( System.err );
        }
        try( OutputStream s = new FileOutputStream( "data/model_q.txt" ) )
        {
            save( model.q, new OutputStreamWriter( s, StandardCharsets.UTF_8 ) );
        }
        catch( IOException ex )
        {
            ex.printStackTrace( System.err );
        }
    }

    private void save( float[][] m, Writer w ) throws IOException
    {
        for( float[] r : m )
        {
            for( float v : r )
                w.write( String.format( "%+21.9f ", v ) );
            w.write( '\n' );
        }
        w.close();
    }

    private float epoch( int sy, int lri )
    {
        float[] y = new float[sy+ model.sb];
        double loss = 0F;
        for( IO p : io )
        {
//            float[] x = p.inp( sb );
            float[] x = p.inp(); // bias included
            float[] h = model.infer( x, y );
//            System.out.printf( "%1.0f%1.0f%1.0f%1.0f %+6.3f %+6.3f %n",
//                    p.inp[3], p.inp[2], p.inp[1], p.inp[0], y[0], y[1] );
//            for( float v : h ) System.out.printf( "%+6.3f ", v ); System.out.println();

            float[] t = p.out();
            float[] e = new float[y.length-model.sb];
            for( int i = 0; i < e.length; i++ )
            {
                e[i] = t[i] - y[i];
                loss += e[i] * e[i]; //TODO max ?
            }
            model.train( x, h, e, lri );
        }
        return (float) Math.sqrt( loss / io.size() );
    }

}
