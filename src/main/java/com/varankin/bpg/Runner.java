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
        runner.run( 0.000750F, 100 );
    }

    private final Model model;
    private final DataSource io;

    private Runner( int sx, int sh, int sy, boolean bias )
    {
        model = new Model( sx, sh, sy, bias );
        io = new DataArray( sx, bias );
    }

    private void run( float accuracy, int lri )
    {
        float loss;
        int n = 0;
        do
        {
            // 4*16*2 schema:
            // *100   Epoch #550331 loss: 0,049
            // *1000  Epoch #2081 loss: 0,041
            // *10000 Epoch #15927 loss: 0,050
            loss = epoch( lri );
            System.out.printf( "Epoch #%d loss: %5.6f%n", n++, loss );
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

    private float epoch( int lri )
    {
        double loss = 0F;
        for( IO p : io )
        {
            model.infer( p.inp() );
//            System.out.printf( "%1.0f%1.0f%1.0f%1.0f %+6.3f %+6.3f %n",
//                    p.inp[3], p.inp[2], p.inp[1], p.inp[0], model.y[0], model.y[1] );
//            for( float v : model.h ) System.out.printf( "%+6.3f ", v ); System.out.println();

            float[] t = p.out();
            float[] e = new float[t.length];
            for( int i = 0; i < e.length; i++ )
            {
                e[i] = t[i] - model.y[i];
                loss += e[i] * e[i]; //TODO max ?
            }
            model.train( e, lri );
        }
        return (float) Math.sqrt( loss / io.size() );
    }

}
