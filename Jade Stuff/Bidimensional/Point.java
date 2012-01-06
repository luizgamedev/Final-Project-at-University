/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package TestBidimensional;

/**
 *
 * @author kazenotenshi
 */
public class Point {

    public int x;
    public int y;

    public Point(){

    }

    public Point(int x, int y){
        this.x = x;
        this.y = y;
    }

    public String toString(){
        return "["+x+","+y+"]";
    }

    public void soma(Point p){
        this.x += p.x;
        this.y += p.y;
    }

    public void soma(int x, int y){
        this.x += x;
        this.y += y;
    }

    public boolean equals(Point a, Point b){
        if(a.x == b.x && a.y == b.y)
            return true;
        else
            return false;
    }
    public boolean equals(Point a){
        if(this.x == a.x && this.y == a.y)
            return true;
        else
            return false;
    }

}
